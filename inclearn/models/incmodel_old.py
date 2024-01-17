import math
import os
import random
import time
from copy import deepcopy
from ignite.utils import to_onehot

import numpy as np
import torch
from torch import nn
from inclearn.convnet import network
from inclearn.convnet.utils import (extract_features, finetune_last_layer,
                                    update_classes_mean)
from inclearn.datasets.dataset import get_dataset
from inclearn.learn.losses import energy_criterion
from inclearn.models.base import IncrementalLearner
from inclearn.tools import factory, utils
from inclearn.tools.memory import MemorySize
from inclearn.tools.metrics import ClassErrorMeter
from inclearn.tools.scheduler import GradualWarmupScheduler
from scipy.spatial.distance import cdist
from torch.nn import DataParallel
from torch.nn import functional as F

# Constants
EPSILON = 1e-8


class IncModel(IncrementalLearner):
    def __init__(self, cfg, trial_i, _run, ex, tensorboard, inc_dataset):
        super().__init__()
        self._cfg = cfg
        self._device = cfg['device']
        self._ex = ex
        self._run = _run  # the sacred _run object.

        # Data
        self._inc_dataset = inc_dataset
        self._n_classes = 0
        self._trial_i = trial_i  # which class order is used

        # Optimizer paras
        self._opt_name = cfg["optimizer"]
        self._warmup = cfg['warmup']
        self._lr = cfg["lr"]
        self._weight_decay = cfg["weight_decay"]
        self._n_epochs = cfg["epochs"]
        self._scheduling = cfg["scheduling"]
        self._lr_decay = cfg["lr_decay"]

        # Classifier Learning Stage
        self._decouple = cfg["decouple"]

        # Logging
        self._tensorboard = tensorboard
        if f"trial{self._trial_i}" not in self._run.info:
            self._run.info[f"trial{self._trial_i}"] = {}
        self._val_per_n_epoch = cfg["val_per_n_epoch"]

        # Model
        self._der = cfg['der']  # Whether to expand the representation
        self._network = network.BasicNet(
            cfg["convnet"],
            cfg=cfg,
            nf=cfg["channel"],
            device=self._device,
            use_bias=cfg["use_bias"],
            dataset=cfg["dataset"],
        )
        self._parallel_network = DataParallel(self._network)
        self._train_head = cfg["train_head"]
        self._infer_head = cfg["infer_head"]
        self._old_model = None

        # Learning
        self._temperature = cfg["temperature"]
        self._distillation = cfg["distillation"]

        # Memory
        self._memory_size = MemorySize(cfg["mem_size_mode"], inc_dataset, cfg["memory_size"],
                                       cfg["fixed_memory_per_cls"])
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]
        
        # Outlier
        self._novelty_detection_method = cfg['novelty_detection']['method']
        self._dataloader_out = torch.utils.data.DataLoader(
            get_dataset(cfg['novelty_detection']['dataset_out'])\
                (cfg['novelty_detection']['data_folder'], train=True),
            batch_size=self._cfg['novelty_detection']['batch_size'], 
            shuffle=False,
            num_workers=self._cfg["workers"], 
            pin_memory=True)
        
        if self._cfg["save_ckpt"]:
            save_path = os.path.join(os.getcwd(), "ckpts", self._cfg["exp"]["name"])
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts", 
                                         self._cfg["exp"]["name"], "mem")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

    def eval(self):
        self._parallel_network.eval()

    def train(self):
        if self._der:
            self._parallel_network.train()
            self._parallel_network.module.convnets[-1].train()
            if self._parallel_network.module.use_calibration_backbone:
                self._parallel_network.module.convnets_calib[-1].train()
            if self._task >= 1:
                for i in range(self._task):
                    self._parallel_network.module.convnets[i].eval()                    
                    if self._parallel_network.module.use_calibration_backbone:
                        self._parallel_network.module.convnets_calib[i].eval()                    
        else:
            self._parallel_network.train()

    def _before_task(self, taski, inc_dataset):
        self._ex.logger.info(f"Begin step {taski}")

        # Update Task info
        self._task = taski
        self._n_classes += self._task_size

        # Memory
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        self._ex.logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self._network.add_classes(self._task_size)
        self._network.task_size = self._task_size
        self.set_optimizer()

    def set_optimizer(self, lr=None):
        if lr is None:
            lr = self._lr

        if self._cfg["dynamic_weight_decay"]:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self._cfg["task_max"] / (self._task + 1)
        else:
            weight_decay = self._weight_decay
        self._ex.logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))

        if self._der and self._task > 0:
            for i in range(self._task):
                for p in self._parallel_network.module.convnets[i].parameters():
                    p.requires_grad = False
                
                if self._parallel_network.module.use_calibration_backbone:
                    for p in self._parallel_network.module.convnets_calib[i].parameters():
                        p.requires_grad = False
                    
                    for p in self._parallel_network.module.calibrated_classifiers[i].parameters():
                        p.requires_grad = False
        params = [{
            'params': filter(lambda p: p.requires_grad, self._network.standard_parameters()),
            'lr': lr
        }]
        if self._cfg["novelty_detection"]["enable"]:
            params.append({
                'params': filter(lambda p: p.requires_grad, self._network.calibration_parameters()),
                'lr': self._cfg["novelty_detection"]["optimizer"]["lr"]
            })
            
        self._optimizer = factory.get_optimizer(
            params,
            self._opt_name, 
            lr, 
            weight_decay
        )

        if "cos" in self._cfg["scheduler"]:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self._n_epochs)
        else:
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   self._scheduling,
                                                                   gamma=self._lr_decay)

        if self._warmup:
            print("warmup")
            self._warmup_scheduler = GradualWarmupScheduler(self._optimizer,
                                                            multiplier=1,
                                                            total_epoch=self._cfg['warmup_epochs'],
                                                            after_scheduler=self._scheduler)

    def _generator_from_loader(self, dataloader):
        while True:
            for images, targets in dataloader:
                yield images, targets

    def _train_task(self, train_loader, val_loader):
        self._ex.logger.info(f"nb {len(train_loader.dataset)}")

        topk = 5 if self._n_classes > 5 else self._task_size
        accu = ClassErrorMeter(accuracy=True, topk=[1, topk])
        train_new_accu = ClassErrorMeter(accuracy=True)
        train_old_accu = ClassErrorMeter(accuracy=True)

        utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "Initial trainset")
        utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Initial trainset")

        self._optimizer.zero_grad()
        self._optimizer.step()

        for epoch in range(self._n_epochs):
            _loss_ce, _loss_aux, _loss_novelty = 0.0, 0.0, 0.0
            accu.reset()
            train_new_accu.reset()
            train_old_accu.reset()
            if self._warmup:
                self._warmup_scheduler.step()
                if epoch == self._cfg['warmup_epochs']:
                    self._network.classifier.reset_parameters()
                    self._network.aux_classifier.reset_parameters()
                    
                    # if self._network.use_calibration_head:
                    #     self._network.calibrated_classifiers.reset_parameters()
            
            self._dataloader_out.dataset.offset = np.random.randint(len(self._dataloader_out.dataset))
            generator_out = self._generator_from_loader(self._dataloader_out)
                        
            for i, (inliers, targets) in enumerate(train_loader, start=1):
                outliers, _ = next(generator_out)
                
                self.train()
                self._optimizer.zero_grad()
                
                new_classes = targets >= (self._n_classes - self._task_size)
                old_classes = targets < (self._n_classes - self._task_size)
                
                loss_ce, loss_aux, loss_novelty = self._forward_loss(
                    inliers,
                    outliers,
                    targets,
                    old_classes,
                    new_classes,
                    accu=accu,
                    new_accu=train_new_accu,
                    old_accu=train_old_accu,
                )
                loss = torch.zeros([1]).to(self._device, non_blocking=True)
                loss += loss_ce
                if self._cfg["use_aux_cls"] and self._task > 0:
                    loss += loss_aux
                    
                if self._cfg["novelty_detection"]["enable"]:
                    loss += loss_novelty

                if torch.isnan(loss):
                    raise RuntimeError('Loss has NaN values')

                loss.backward()
                self._optimizer.step()

                if self._cfg["postprocessor"]["enable"]:
                    if self._cfg["postprocessor"]["type"].lower() == "wa":
                        for p in self._network.classifier.parameters():
                            p.data.clamp_(0.0)
                        
                        # Not Tested
                        if self._network.use_calibration_head:
                            for p in self._network.calibrated_classifiers.parameters():
                                p.data.clamp_(0.0)

                _loss_ce += loss_ce
                _loss_aux += loss_aux
                _loss_novelty += loss_novelty
                
            _loss_ce = _loss_ce.item()
            _loss_aux = _loss_aux.item()
            _loss_novelty = _loss_novelty.item()
            
            if not self._warmup:
                self._scheduler.step()
            self._ex.logger.info(
                "Task {}/{}, Epoch {}/{} => Clf loss: {} Aux loss: {} Novelty loss: {}, Train Accu: {}, Train@5 Acc: {}, old acc:{}".
                format(
                    self._task + 1,
                    self._n_tasks,
                    epoch + 1,
                    self._n_epochs,
                    round(_loss_ce / i, 3),
                    round(_loss_aux / i, 3),
                    round(_loss_novelty / i, 3),
                    round(accu.value()[0], 3),
                    round(accu.value()[1], 3),
                    round(train_old_accu.value()[0], 3),
                )
            )

            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

        # For the large-scale dataset, we manage the data in the shared memory.
        self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory

        utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "After training")
        utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Trainset")
        self._run.info[f"trial{self._trial_i}"][f"task{self._task}_train_accu"] = round(accu.value()[0], 3)

    def _forward_loss(self, inliers, outliers, targets, old_classes, new_classes, accu=None, new_accu=None, old_accu=None):
        
        inliers = inliers.to(self._device, non_blocking=True)
        outliers = outliers.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)

        inputs = torch.cat((inliers, outliers), 0)
        outputs = self._parallel_network(inputs) 
        loss_novelty = self._compute_novelty_loss(outputs, inliers)

        outputs_inlier = outputs
        for key, value in outputs_inlier.items():
            if value is not None:
                outputs_inlier[key] = value[:len(inliers)]

        if accu is not None:
            accu.add(outputs_inlier['logit'], targets)
        
        loss_ce, loss_aux, loss_ce_calibrated = self._compute_cls_loss(targets, outputs_inlier, 
                                                   old_classes, new_classes)

        loss_novelty += loss_ce_calibrated

        return loss_ce, loss_aux, loss_novelty

    def _compute_novelty_loss(self, outputs, inliers):
        if self._cfg['novelty_detection']['calibration_head']:
            output_logits = outputs['calibrated_logit']
        else:
            output_logits = outputs['logit']
            
        inlier_logits = output_logits[:len(inliers)]
        outlier_logits = output_logits[len(inliers):]
        
        m_in = self._cfg['novelty_detection']['m_in']
        m_out = self._cfg['novelty_detection']['m_out']
        
        return energy_criterion(inlier_logits, outlier_logits, m_in, m_out)

    def _compute_cls_loss(self, targets, outputs, old_classes, new_classes):
        loss_ce = F.cross_entropy(outputs['logit'], targets)
        
        loss_ce_calibrated = torch.zeros([1]).cuda()
        if outputs['calibrated_logit'] is not None:
            loss_ce_calibrated = F.cross_entropy(outputs['calibrated_logit'], targets)

        loss_aux = torch.zeros([1]).cuda()
        if outputs['aux_logit'] is not None:
            aux_targets = targets.clone()
            if self._cfg["aux_n+1"]:
                aux_targets[old_classes] = 0
                aux_targets[new_classes] -= sum(self._inc_dataset.increments[:self._task]) - 1
            loss_aux = F.cross_entropy(outputs['aux_logit'], aux_targets)
            
        return loss_ce, loss_aux, loss_ce_calibrated

    def _after_task(self, taski, inc_dataset):
        network = deepcopy(self._parallel_network)
        network.eval()
        self._ex.logger.info("save model")
        if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
            save_path = os.path.join(os.getcwd(), "ckpts", self._cfg["exp"]["name"])
            torch.save(network.cpu().state_dict(), "{}/step{}.ckpt".format(save_path, self._task))

        if (self._cfg["decouple"]['enable'] and taski > 0):
            if self._cfg["decouple"]["fullset"]:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
            else:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                       inc_dataset.targets_inc,
                                                       mode="balanced_train")

            # finetuning
            self._parallel_network.module.classifier.reset_parameters()
            self.finetune_last_layer(train_loader,
                                nepoch=self._decouple["epochs"],
                                loss_type="ce")
            
            network = deepcopy(self._parallel_network)
            if self._cfg["save_ckpt"]:
                save_path = os.path.join(os.getcwd(), "ckpts", self._cfg["exp"]["name"])
                os.makedirs(save_path, exist_ok=True)
                torch.save(network.cpu().state_dict(), 
                           "{}/decouple_step{}.ckpt".format(save_path, self._task))

        if self._cfg["postprocessor"]["enable"]:
            self._update_postprocessor(inc_dataset)

        if self._cfg["infer_head"] == 'NCM':
            self._ex.logger.info("compute prototype")
            self.update_prototype()

        if self._memory_size.memsize != 0:
            self._ex.logger.info("build memory")
            self.build_exemplars(inc_dataset, self._coreset_strategy)

            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts", self._cfg["exp"]["name"], "mem")
                os.makedirs(save_path, exist_ok=True)
                memory = {
                    'x': inc_dataset.data_memory,
                    'y': inc_dataset.targets_memory,
                    'herding': self._herding_matrix
                }
                if not (os.path.exists(f"{save_path}/mem_step{self._task}.ckpt") and self._cfg['load_mem']):
                    torch.save(memory, "{}/mem_step{}.ckpt".format(save_path, self._task))
                    self._ex.logger.info(f"Save step{self._task} memory!")

        self._parallel_network.eval()
        self._old_model = deepcopy(self._parallel_network)
        self._old_model.module.freeze()
        del self._inc_dataset.shared_data_inc
        self._inc_dataset.shared_data_inc = None

    def finetune_last_layer(
        self,
        loader,
        nepoch=30,
        loss_type="ce",
        test_loader=None
    ):
        self._parallel_network.eval()
        params = [{
            'params': self._parallel_network.module.classifier.parameters(),
            'lr': self._decouple["lr"]
        }]
        if self._parallel_network.module._cfg['novelty_detection']['calibration_head']:
            params.append({
                'params': self._parallel_network.module.calibrated_classifiers[-1].parameters(),
                'lr': self._cfg["novelty_detection"]["optimizer"]["lr"]
            })
        
        optim = factory.get_optimizer(
            params,
            self._decouple["optimizer"], 
            self._decouple["lr"], 
            self._decouple["weight_decay"]
        )
        
        # optim = SGD(network.module.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, 
            self._decouple["scheduling"], 
            gamma=self._decouple["lr_decay"])

        if loss_type == "ce":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        self._ex.logger.info("Begin finetuning last layer")

        for i in range(nepoch):
            total_ce_loss = 0.0
            total_novelty_loss = 0.0
            total_correct = 0.0
            total_inlier_count = 0
            total_data_count = 0
            # print(f"dataset loader length {len(loader.dataset)}")
            
            self._dataloader_out.dataset.offset = np.random.randint(len(self._dataloader_out.dataset))
            generator_out = self._generator_from_loader(self._dataloader_out)
            for inliers, targets in loader:
                outliers, _ = next(generator_out)
                inliers, targets, outliers = inliers.cuda(), targets.cuda(), outliers.cuda()
                inputs = torch.cat((inliers, outliers), 0)
                
                if loss_type == "bce":
                    targets = to_onehot(targets, self._n_classes)
                outputs = self._parallel_network(inputs)
                logits = outputs['logit']
                logits_inlier = logits[:len(inliers)]
                
                _, preds = logits_inlier.max(1)
                optim.zero_grad()
                loss_ce = criterion(logits_inlier / self._decouple["temperature"], targets)
                
                if self._cfg['novelty_detection']['calibration_head']:
                    calibration_logits = outputs['calibrated_logit']
                else:
                    calibration_logits = outputs['logit']
                
                calibration_inlier = calibration_logits[:len(inliers)]
                calibration_outlier = calibration_logits[len(inliers):]
                
                m_in = self._cfg['novelty_detection']['m_in']
                m_out = self._cfg['novelty_detection']['m_out']
                loss_novelty = energy_criterion(calibration_inlier, calibration_outlier, m_in, m_out)
                
                loss = torch.zeros([1]).cuda()
                loss += loss_ce
                
                if self._cfg["novelty_detection"]["enable"]:
                    loss += loss_novelty
                
                loss.backward()
                optim.step()
                
                total_ce_loss += loss_ce.detach().cpu().numpy() * inliers.size(0)
                total_novelty_loss += loss_novelty.detach().cpu().numpy() * inputs.size(0)
                total_correct += (preds == targets).sum()
                total_inlier_count += inliers.size(0)
                total_data_count += inputs.size(0)

            if test_loader is not None:
                test_correct = 0.0
                test_count = 0.0
                with torch.no_grad():
                    for inliers, targets in test_loader:
                        outputs = self._parallel_network(inliers.cuda())['logit']
                        _, preds = outputs.max(1)
                        test_correct += (preds.cpu() == targets).sum().item()
                        test_count += inliers.size(0)

            scheduler.step()
            if test_loader is not None:
                self._ex.logger.info(
                    "Epoch %d finetuning CE loss %.3f, Novelty loss %.3f, acc %.3f, Eval %.3f" %
                    (i, 
                    total_ce_loss.item() / total_inlier_count, 
                    total_novelty_loss.item() / total_data_count, 
                    total_correct.item() / total_inlier_count, 
                    test_correct / test_count))
            else:
                self._ex.logger.info("Epoch %d finetuning CE loss, Novelty loss %.3f, %.3f acc %.3f" %
                            (i, 
                            total_ce_loss.item() / total_inlier_count, 
                            total_novelty_loss.item() / total_data_count,
                            total_correct.item() / total_inlier_count))

    def _eval_task(self, data_loader):
        if self._infer_head == "softmax":
            ypred, ytrue = self._compute_accuracy_by_netout(data_loader)
        elif self._infer_head == "NCM":
            ypred, ytrue = self._compute_accuracy_by_ncm(data_loader)
        else:
            raise ValueError()

        return ypred, ytrue

    def _compute_accuracy_by_netout(self, data_loader):
        preds, targets = [], []
        self._parallel_network.eval()
        with torch.no_grad():
            for i, (inputs, lbls) in enumerate(data_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                _preds = self._parallel_network(inputs)['logit']
                if self._cfg["postprocessor"]["enable"] and self._task > 0:
                    _preds = self._network.postprocessor.post_process(_preds, self._task_size)
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        return preds, targets

    def _compute_accuracy_by_ncm(self, loader):
        features, targets_ = extract_features(self._parallel_network, loader)
        targets = np.zeros((targets_.shape[0], self._n_classes), np.float32)
        targets[range(len(targets_)), targets_.astype("int32")] = 1.0

        class_means = (self._class_means.T / (np.linalg.norm(self._class_means.T, axis=0) + EPSILON)).T

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
        # Compute score for iCaRL
        sqd = cdist(class_means, features, "sqeuclidean")
        score_icarl = (-sqd).T
        return score_icarl[:, :self._n_classes], targets_

    def _update_postprocessor(self, inc_dataset):
        if self._cfg["postprocessor"]["type"].lower() == "bic":
            if self._cfg["postprocessor"]["disalign_resample"] is True:
                bic_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                     inc_dataset.targets_inc,
                                                     mode="train",
                                                     resample='disalign_resample')
            else:
                xdata, ydata = inc_dataset._select(inc_dataset.data_train,
                                                   inc_dataset.targets_train,
                                                   low_range=0,
                                                   high_range=self._n_classes)
                bic_loader = inc_dataset._get_loader(xdata, ydata, shuffle=True, mode='train')
            bic_loss = None
            self._network.postprocessor.reset(n_classes=self._n_classes)
            self._network.postprocessor.update(self._ex.logger,
                                               self._task_size,
                                               self._parallel_network,
                                               bic_loader,
                                               loss_criterion=bic_loss)
        elif self._cfg["postprocessor"]["type"].lower() == "wa":
            self._ex.logger.info("Post processor wa update !")
            self._network.postprocessor.update(self._network.classifier, self._task_size)

    def update_prototype(self):
        if hasattr(self._inc_dataset, 'shared_data_inc'):
            shared_data_inc = self._inc_dataset.shared_data_inc
        else:
            shared_data_inc = None
        self._class_means = update_classes_mean(self._parallel_network,
                                                self._inc_dataset,
                                                self._n_classes,
                                                self._task_size,
                                                share_memory=self._inc_dataset.shared_data_inc,
                                                metric='None')

    def build_exemplars(self, inc_dataset, coreset_strategy):
        save_path = os.path.join(os.getcwd(), "ckpts", 
                                 self._cfg["exp"]["name"], 
                                 f"/mem/mem_step{self._task}.ckpt")
        if self._cfg["load_mem"] and os.path.exists(save_path):
            memory_states = torch.load(save_path)
            self._inc_dataset.data_memory = memory_states['x']
            self._inc_dataset.targets_memory = memory_states['y']
            self._herding_matrix = memory_states['herding']
            self._ex.logger.info(f"Load saved step{self._task} memory!")
            return

        if coreset_strategy == "random":
            from inclearn.tools.memory import random_selection

            self._inc_dataset.data_memory, self._inc_dataset.targets_memory = random_selection(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._ex.logger,
                inc_dataset,
                self._memory_per_class,
            )
        elif coreset_strategy == "iCaRL":
            from inclearn.tools.memory import herding
            data_inc = self._inc_dataset.shared_data_inc if self._inc_dataset.shared_data_inc is not None else self._inc_dataset.data_inc
            self._inc_dataset.data_memory, self._inc_dataset.targets_memory, self._herding_matrix = herding(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._herding_matrix,
                inc_dataset,
                data_inc,
                self._memory_per_class,
                self._ex.logger,
            )
        else:
            raise ValueError()

    def validate(self, data_loader):
        if self._infer_head == 'NCM':
            self.update_prototype()
        ypred, ytrue = self._eval_task(data_loader)
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        self._ex.logger.info(f"test top1acc:{test_acc_stats['top1']}")
