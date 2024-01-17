import math
import os
import random
import time
from copy import deepcopy
from ignite.utils import to_onehot

import numpy as np
import torch
from torch import nn
from inclearn.convnet import network_pruning as network
from inclearn.convnet.utils import (extract_features, finetune_last_layer, update_exponential_moving_average, tensor_is_in,
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
        
        self._optimizers = []
        self._schedulers = []
        self._warmup_schedulers = []
        self._num_heads = 0
        self.batch_buffers = []
        
        self._latest_task = None
        self._classes_in_task = []
        self._task_sizes = []

        # Memory
        self._memory_size = MemorySize(cfg["mem_size_mode"], inc_dataset, cfg["memory_size"],
                                       cfg["fixed_memory_per_cls"])
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]
        
        # Outlier and novelty thresholds
        self._novelty_detection_method = cfg['novelty_detection']['method']
        self._dataloader_out = torch.utils.data.DataLoader(
            get_dataset(cfg['novelty_detection']['dataset_out'])\
                (cfg['novelty_detection']['data_folder'], train=True),
            batch_size=self._cfg['novelty_detection']['batch_size'], 
            shuffle=False,
            num_workers=self._cfg["workers"], 
            pin_memory=True
        )
        self.curr_iter = 0

        # Pruning
        self.enable_pruning = self._cfg['pruning']['enable']
        self.lambda_pruning = self._cfg['pruning']['lambda']
        
        self.num_samples = 0.0
        
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
            
            for i in range(self._num_heads):
                if i != self._network.active_head:
                    self._network.convnets[i].eval()
                    if self._network.use_calibration_backbone:
                        self._network.convnets_calib[i].eval()
                else:
                    self._network.convnets[i].train()
                    if self._network.use_calibration_backbone:
                        self._network.convnets_calib[i].train()
        else:
            self._parallel_network.train()

    def _before_task(self, taski, inc_dataset, train_loader=None):
        
        # Update Task info
        self._task = taski
        if self._latest_task is None or taski > self._latest_task:
            self._latest_task = taski
            self._n_classes += self._task_size
            
            self._classes_in_task.append([inc_dataset.classes_cur])
        
        if len(self._network.novelty_means) != 0:
        # if train_loader is not None or len(self._network.novelty_avgs) != 0:
            # Decide which head to train
            self._ex.logger.info(f"Checking novelty for step {taski}")
            
            novelty_means = 0.0
            n = 0.0
            for features, _ in train_loader:
                with torch.no_grad():
                    outputs = self._parallel_network(features)

                    curr_scores = outputs["novelty_scores"]
                    curr_means = torch.mean(curr_scores, dim=0)
                    
                    novelty_means = novelty_means + curr_means
                    n += 1
            
            novelty_means = novelty_means / n
            novelty_threshold = self._network.novelty_means
            is_novel = novelty_means > novelty_threshold
            
            self._ex.logger.info(f"Current task has novelty: {novelty_means}")
            self._ex.logger.info(f"Thres.: {novelty_threshold}")
            self._ex.logger.info(f"Center.: {self._network.novelty_means}")
            self._ex.logger.info(f"Std.: {torch.sqrt(self._network.novelty_vars)}")
            self._ex.logger.info(f"Novelty decisions: {is_novel}")
            
            if is_novel.sum() == novelty_means.shape[-1]:
                # New task, grow new head
                self._network.active_head = novelty_means.shape[-1]
                self.grow_head()
                
                # Set current head to train
                self._ex.logger.info(f"Training new head {self._network.active_head}")
            else:
                self._network.active_head = torch.argmin(novelty_means - novelty_threshold)
                self.update_existing_head()
                
                self._ex.logger.info(f"Retraining head {self._network.active_head}")
        else:
            # No tasks trained
            self._ex.logger.info(f"Begin step {taski}")
            self._network.active_head = 0
            self.grow_head()
            self._ex.logger.info(f"Training new head {self._network.active_head}")
        
        self._network.active_head = self._network.active_head
        
    # def reset_novelty_thresholds(self):
    #     self.novelty_avg.data = torch.tensor(0.0).to(self._device)
    #     self.novelty_var.data = torch.tensor(0.0).to(self._device)
    #     self.num_samples = 0

    def update_existing_head(self):
        self._ex.logger.info(f"Updating existing head")        
        
    def grow_head(self):
        self._ex.logger.info(f"Growing new head")

        # Update Task info
        self._task_sizes.append(self._task_size)
        self._num_heads += 1
        
        # Grow memory
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        self._ex.logger.info("Now {} examplars per class.".format(self._memory_per_class))

        # Grow network
        self._network.add_classes(self._task_size)
        self._network.task_size = self._task_size
        
        # Grow optimizers
        optimizers, schedulers, warmup_schedulers = self.get_new_optimizers()
        self._optimizers.append(optimizers)
        self._schedulers.append(schedulers)
        self._warmup_schedulers.append(warmup_schedulers)
        
        # Grow novelty stats
        self._network.grow_novelty_stats()
        self.batch_buffers.append(None)
        
    def get_head_params_with_index(self, idx):
        if idx == -1:
            idx = self._num_heads - 1

        for i in range(self._num_heads):
            for p in self._network.convnets[i].parameters():
                p.requires_grad = True
            
            if self._network.use_calibration_head and \
                self._network.use_multihead_calibration:
                for p in self._network.calibrated_classifiers[i].parameters():
                    p.requires_grad = True
            
            if self._network.use_calibration_backbone:
                for p in self._network.convnets_calib[i].parameters():
                    p.requires_grad = True
        
        for i in range(self._num_heads):
            if i != idx:
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False
                
                if self._network.use_calibration_head and \
                    self._network.use_multihead_calibration:
                    for p in self._network.calibrated_classifiers[i].parameters():
                        p.requires_grad = False
                
                if self._network.use_calibration_backbone:
                    for p in self._network.convnets_calib[i].parameters():
                        p.requires_grad = False
        
        standard_params = filter(lambda p: p.requires_grad, self._network.standard_parameters())
        calib_params = filter(lambda p: p.requires_grad, self._network.calibration_parameters())
        
        # for p in self._network.parameters():
        #     p.requires_grad = True
        
        return standard_params, calib_params
    
    def get_new_optimizers(self):
        lr = self._lr

        if self._cfg["dynamic_weight_decay"]:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self._cfg["task_max"] / (self._num_heads + 1)
        else:
            weight_decay = self._weight_decay
        self._ex.logger.info("Step {} weight decay {:.5f}".format(self._num_heads, weight_decay))
        
        standard_params, calib_params = self.get_head_params_with_index(self._network.active_head)
        
        optimizers = {}
        schedulers = {}
        
        optimizer_base = factory.get_optimizer(
            standard_params,
            self._opt_name, 
            lr, 
            weight_decay
        )
        if "cos" in self._cfg["scheduler"]:
            scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, self._n_epochs)
        else:
            scheduler_base = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_base,
                self._scheduling,
                gamma=self._lr_decay
            )
        optimizers['optimizer_base'] = optimizer_base
        schedulers['scheduler_base'] = scheduler_base
        
        if self._cfg["novelty_detection"]["enable"]:
            optimizer_novelty = factory.get_optimizer(
                calib_params,
                self._opt_name,
                self._cfg["novelty_detection"]["optimizer"]["lr"],
                self._cfg["novelty_detection"]["optimizer"]["weight_decay"]
            )
            if "cos" in self._cfg["scheduler"]:
                scheduler_novelty = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_novelty, 
                    self._n_epochs
                )
            else:
                scheduler_novelty = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer_novelty,
                    self._scheduling,
                    gamma=self._lr_decay
                )
            optimizers['optimizer_novelty'] = optimizer_novelty
            schedulers['scheduler_novelty'] = scheduler_novelty

        warmup_schedulers = {}
        if self._warmup:
            print("warmup")
            warmup_scheduler_base = GradualWarmupScheduler(
                optimizer_base,
                multiplier=1,
                total_epoch=self._cfg['warmup_epochs'],
                after_scheduler=scheduler_base
            )
            warmup_schedulers['scheduler_base'] = warmup_scheduler_base

            if self._cfg["novelty_detection"]["enable"]:
                warmup_scheduler_novelty = GradualWarmupScheduler(
                    optimizer_novelty,
                    multiplier=1,
                    total_epoch=self._cfg['warmup_epochs'],
                    after_scheduler=scheduler_novelty
                )
                warmup_schedulers['scheduler_novelty'] = warmup_scheduler_novelty
                
        return optimizers, schedulers, warmup_schedulers

    def freeze_backbone_excluding_index(self, idx):
        for i in range(self._num_heads):
            for p in self._network.convnets[i].parameters():
                p.requires_grad = True
            
            if self._network.use_calibration_head and \
                self._network.use_multihead_calibration:
                for p in self._network.calibrated_classifiers[i].parameters():
                    p.requires_grad = True
            
            if self._network.use_calibration_backbone:
                for p in self._network.convnets_calib[i].parameters():
                    p.requires_grad = True
        
        for i in range(self._num_heads):
            if i != idx:
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False
                
                if self._network.use_calibration_head and \
                    self._network.use_multihead_calibration:
                    for p in self._network.calibrated_classifiers[i].parameters():
                        p.requires_grad = False
                
                if self._network.use_calibration_backbone:
                    for p in self._network.convnets_calib[i].parameters():
                        p.requires_grad = False
        

    # def set_optimizer(self, lr=None):
    #     if lr is None:
    #         lr = self._lr

    #     if self._cfg["dynamic_weight_decay"]:
    #         # used in BiC official implementation
    #         weight_decay = self._weight_decay * self._cfg["task_max"] / (self._task + 1)
    #     else:
    #         weight_decay = self._weight_decay
    #     self._ex.logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))

    #     if self._der and self._task > 0:
    #         for i in range(self._task):
    #             for p in self._network.convnets[i].parameters():
    #                 p.requires_grad = False
                
    #             if self._network.use_calibration_head and \
    #                 self._network.use_multihead_calibration:
    #                 for p in self._network.calibrated_classifiers[i].parameters():
    #                     p.requires_grad = False
                
    #             if self._network.use_calibration_backbone:
    #                 for p in self._network.convnets_calib[i].parameters():
    #                     p.requires_grad = False
        
    #     self._optimizers = {}
    #     self._schedulers = {}
        
    #     optimizer_base = factory.get_optimizer(
    #         filter(lambda p: p.requires_grad, self._network.standard_parameters()),
    #         self._opt_name, 
    #         lr, 
    #         weight_decay
    #     )
    #     if "cos" in self._cfg["scheduler"]:
    #         scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, self._n_epochs)
    #     else:
    #         scheduler_base = torch.optim.lr_scheduler.MultiStepLR(
    #             optimizer_base,
    #             self._scheduling,
    #             gamma=self._lr_decay
    #         )
    #     self._optimizers['optimizer_base'] = optimizer_base
    #     self._schedulers['scheduler_base'] = scheduler_base
        
    #     if self._cfg["novelty_detection"]["enable"]:
    #         optimizer_novelty = factory.get_optimizer(
    #             filter(lambda p: p.requires_grad, self._network.calibration_parameters()),
    #             self._opt_name,
    #             self._cfg["novelty_detection"]["optimizer"]["lr"],
    #             self._cfg["novelty_detection"]["optimizer"]["weight_decay"]
    #         )
    #         if "cos" in self._cfg["scheduler"]:
    #             scheduler_novelty = torch.optim.lr_scheduler.CosineAnnealingLR(
    #                 optimizer_novelty, 
    #                 self._n_epochs
    #             )
    #         else:
    #             scheduler_novelty = torch.optim.lr_scheduler.MultiStepLR(
    #                 optimizer_novelty,
    #                 self._scheduling,
    #                 gamma=self._lr_decay
    #             )
    #         self._optimizers['optimizer_novelty'] = optimizer_novelty
    #         self._schedulers['scheduler_novelty'] = scheduler_novelty

    #     if self._warmup:
    #         print("warmup")
    #         self._warmup_schedulers = {}
    #         warmup_scheduler_base = GradualWarmupScheduler(
    #             optimizer_base,
    #             multiplier=1,
    #             total_epoch=self._cfg['warmup_epochs'],
    #             after_scheduler=scheduler_base
    #         )
    #         self._warmup_schedulers['scheduler_base'] = warmup_scheduler_base

    #         if self._cfg["novelty_detection"]["enable"]:
    #             warmup_scheduler_novelty = GradualWarmupScheduler(
    #                 optimizer_novelty,
    #                 multiplier=1,
    #                 total_epoch=self._cfg['warmup_epochs'],
    #                 after_scheduler=scheduler_novelty
    #             )
    #             self._warmup_schedulers['scheduler_novelty'] = warmup_scheduler_novelty
    
    # 0 1 --> backbone0
    # 0 1 2 3 --> 0 1 backbone0 (novelty detector), 2 3 backbone1 (novelty detector)
    # 5 Backbones
    
    
    # DER
    # 0 1 --> backbone0 (1000 #data points)
    # 0 1 2 3 --> backbone1 (1000 #data points)
    # 2 3 --> backbone2 (1000 #data points)
    # 
    # Memory Large: 10 Backbones
    
    # 0 1 2 3 4 (20%)
    # 0 1 2 3 4 5 6 7 8 9 --> all classes (20%)
    # 5 6 7 8 9 (20%)

    def _generator_from_loader(self, dataloader):
        while True:
            for images, targets in dataloader:
                yield images, targets

    def _train_task(self, train_loader, val_loader):
        # No heads exist grow new head
        if len(self._network.novelty_means) == 0:
            self._network.active_head = 0
            self.grow_head()
            self._ex.logger.info(f"Growing new head {self._network.active_head}")
            
            for features, _ in train_loader:
                with torch.no_grad():
                    outputs = self._parallel_network(features)
                    
                    energy_scores = outputs["novelty_scores"]
                    curr_means = torch.mean(energy_scores, dim=0)
                    curr_vars = torch.var(energy_scores, dim=0)
                    
                    novelty_means = novelty_means + curr_means
                    novelty_vars = novelty_vars + curr_vars
                    n += 1
            
            novelty_means = novelty_means / n
            novelty_vars = novelty_vars / n 
            
            self._network.update_running_novelty_stats(novelty_means, novelty_vars)
            self._network.update_base_novelty_stats(novelty_means, novelty_vars)
        
        self._ex.logger.info(f"nb {len(train_loader.dataset)}")
        
        train_new_accu = ClassErrorMeter(accuracy=True)
        train_old_accu = ClassErrorMeter(accuracy=True)
        
        utils.display_weight_norm(
            self._ex.logger, 
            self._parallel_network, 
            self._increments, 
            "Initial trainset"
        )
        utils.display_feature_norm(
            self._ex.logger, 
            self._parallel_network, 
            train_loader, 
            self._n_classes,
            self._increments, 
            "Initial trainset"
        )
        
        #######################################
        
        for epoch in range(self._n_epochs):
            _loss_ce, _loss_aux, _loss_novelty, _loss_pruning = 0.0, 0.0, 0.0, 0.0
            accu.reset()
            train_new_accu.reset()
            train_old_accu.reset()
            
            if self._warmup:
                for warmup_schedule in warmup_schedules.values():
                    warmup_schedule.step()
                # if epoch == self._cfg['warmup_epochs']:
                #     self._network.classifier.reset_parameters()
                #     self._network.aux_classifier.reset_parameters()
                #     # if self._network.use_calibration_head:
                #     #     self._network.calibrated_classifiers.reset_parameters()
            
            self._dataloader_out.dataset.offset = np.random.randint(len(self._dataloader_out.dataset))
            generator_out = self._generator_from_loader(self._dataloader_out)

            for optimizer in optimizers.values():
                optimizer.zero_grad()
                optimizer.step()
            
            classes_in_head = self._classes_in_task[self._task]
            for i, (inliers, targets) in enumerate(train_loader, start=1):
                outliers, _ = next(generator_out)
                
                # Determine which expert to train
                self.eval()
                with torch.no_grad():
                    outputs = self._parallel_network(inliers)
                    energy_scores = outputs["novelty_scores"]
                
                means = self._network.novelty_means
                variances = self._network.novelty_vars
                means = torch.cat([means, self._network.base_novelty_mean])
                variances = torch.cat([variances, self._network.base_novelty_var])
                
                novelty_scores = []
                for mean, variance in zip(means, variances):
                    normal_dist = torch.distributions.Normal(mean, variance)
                    expert_novelty_scores = normal_dist.log_prob(energy_scores)
                    novelty_scores.append(expert_novelty_scores)
                novelty_scores = torch.stack([novelty_scores])
                
                expert_assignments = torch.argmax(novelty_scores)
                if any(expert_assignments != expert_assignments[0]):
                    self._ex.logger.info(f"Warning: data classified to different experts")
                
                if torch.max(expert_assignments) >= len(self.batch_buffers):
                    self.grow_head()
                    self._ex.logger.info(f"Growing new head {self._network.active_head}")

                backprop_batches = []
                batch_size = train_loader.batch_size
                for expert_id, buffer in enumerate(self.batch_buffers):
                    inliers_assigned_to_expert = inliers[expert_assignments == expert_id]
                    num_inliers_to_backprop = max(0, len(self.batch_buffers[expert_id]) - batch_size)
                    inliers_backproped = inliers_assigned_to_expert[:num_inliers_to_backprop]
                    inliers_buffered = inliers_assigned_to_expert[num_inliers_to_backprop:]
                    
                    if len(inliers_backproped) > 0:
                        backprop_batch = torch.cat([buffer, inliers_backproped])
                        backprop_batches.append((backprop_batch, expert_id))
                        
                        self.batch_buffers[expert_id] = inliers_buffered
                    else:
                        self.batch_buffers[expert_id] = torch.cat([buffer, inliers_buffered])
                
                # Backprop if any backprop batches exist
                if backprop_batches:
                    self.train()
                    inhead_classes = tensor_is_in(targets, classes_in_head)
                    outhead_classes = torch.logical_not(inhead_classes)
                    
                    optimizer_base = optimizers['optimizer_base']
                    
                    loss = torch.zeros([1]).to(self._device, non_blocking=True)
                    loss_base = torch.zeros([1]).to(self._device, non_blocking=True)
                    loss_ce, loss_aux, loss_novelty = self._forward_loss(
                        inliers,
                        outliers,
                        targets,
                        outhead_classes,
                        inhead_classes,
                        accu=accu,
                        new_accu=train_new_accu,
                        old_accu=train_old_accu,
                    )
                    self.curr_iter += 1
                    
                    loss = loss + loss_ce
                    loss_base = loss_base + loss_ce
                    
                    if self._cfg["use_aux_cls"] and self._num_heads > 1:
                        loss = loss + loss_aux
                        loss_base = loss_base + loss_aux

                    if self.enable_pruning:
                        regs = self._network.regularizer()
                        loss_pruning = regs[self._network.active_head]
                        loss_base = loss_base + self.lambda_pruning * loss_pruning
                    
                    loss_base.backward()
                    optimizer_base.step()
                    
                    for optimizer in optimizers.values():
                        optimizer.zero_grad()
                        optimizer.step()
                    
                    if self._cfg["novelty_detection"]["enable"]:
                        optimizer_novelty = optimizers['optimizer_novelty']
                        loss = loss + loss_novelty
                        loss_novelty.backward()
                        optimizer_novelty.step()
                        
                        for optimizer in optimizers.values():
                            optimizer.zero_grad()
                            optimizer.step()

                    if torch.isnan(loss):
                        raise RuntimeError('Loss has NaN values')

                    if self._cfg["postprocessor"]["enable"]:
                        if self._cfg["postprocessor"]["type"].lower() == "wa":
                            for p in self._network.classifier.parameters():
                                p.data.clamp_(0.0)
                            
                            # Not Tested
                            if self._network.use_calibration_head:
                                for p in self._network.calibrated_classifiers.parameters():
                                    p.data.clamp_(0.0)

                    _loss_ce = _loss_ce + loss_ce
                    _loss_aux = _loss_aux + loss_aux
                    _loss_novelty = _loss_novelty + loss_novelty
                    _loss_pruning = _loss_pruning + loss_pruning
                
            _loss_ce = _loss_ce.item()
            _loss_aux = _loss_aux.item()
            _loss_novelty = _loss_novelty.item()
            _loss_pruning = _loss_pruning.item()
            
            if not self._warmup:
                for scheduler in schedulers.values():
                    scheduler.step()
            
            self._ex.logger.info(
                "Task {}/{}, Epoch {}/{} => Clf loss: {} Aux loss: {} Novelty loss: {}, Pruning loss: {}, Train Accu: {}, Train@5 Acc: {}, old acc:{}".
                format(
                    self._task + 1,
                    self._n_tasks,
                    epoch + 1,
                    self._n_epochs,
                    round(_loss_ce / i, 3),
                    round(_loss_aux / i, 3),
                    round(_loss_novelty / i, 3),
                    round(_loss_pruning / i, 3),
                    round(accu.value()[0], 3),
                    round(accu.value()[1], 3),
                    round(train_old_accu.value()[0], 3),
                ))

            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

        # For the large-scale dataset, we manage the data in the shared memory.
        self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory

        utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "After training")
        utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Trainset")
        self._run.info[f"trial{self._trial_i}"][f"task{self._task}_train_accu"] = round(accu.value()[0], 3)
        
        #########################################
        
        # #######################
        ### Setup to train coresponding expert
        self.freeze_backbone_excluding_index(self._network.active_head)
        topk = 5 if self._n_classes > 5 else self._task_size
        accu = ClassErrorMeter(accuracy=True, topk=[1, topk])

        optimizers = self._optimizers[self._network.active_head]
        schedulers = self._schedulers[self._network.active_head]
        warmup_schedules = self._warmup_schedulers[self._network.active_head]
        ### End setup to train corresponding expert
        
        for optimizer in optimizers.values():
            optimizer.zero_grad()
            optimizer.step()
        
        for epoch in range(self._n_epochs):
            _loss_ce, _loss_aux, _loss_novelty, _loss_pruning = 0.0, 0.0, 0.0, 0.0
            accu.reset()
            train_new_accu.reset()
            train_old_accu.reset()
            
            if self._warmup:
                for warmup_schedule in warmup_schedules.values():
                    warmup_schedule.step()
                # if epoch == self._cfg['warmup_epochs']:
                #     self._network.classifier.reset_parameters()
                #     self._network.aux_classifier.reset_parameters()
                    
                #     # if self._network.use_calibration_head:
                #     #     self._network.calibrated_classifiers.reset_parameters()
            
            self._dataloader_out.dataset.offset = np.random.randint(len(self._dataloader_out.dataset))
            generator_out = self._generator_from_loader(self._dataloader_out)

            for optimizer in optimizers.values():
                optimizer.zero_grad()
                optimizer.step()
            
            
            zero = torch.tensor([0.0])
            self._network.update_running_novelty_stats(zero, zero)
            self.num_samples = 0.0
            classes_in_head = self._classes_in_task[self._task]

            for i, (inliers, targets) in enumerate(train_loader, start=1):
                outliers, _ = next(generator_out)
                
                self.train()
                inhead_classes = tensor_is_in(targets, classes_in_head)
                outhead_classes = torch.logical_not(inhead_classes)
                
                # inhead_classes = targets >= (self._n_classes - self._task_size)
                # outhead_classes = targets < (self._n_classes - self._task_size)
                
                optimizer_base = optimizers['optimizer_base']
                
                loss = torch.zeros([1]).to(self._device, non_blocking=True)
                loss_base = torch.zeros([1]).to(self._device, non_blocking=True)
                loss_ce, loss_aux, loss_novelty = self._forward_loss(
                    inliers,
                    outliers,
                    targets,
                    outhead_classes,
                    inhead_classes,
                    accu=accu,
                    new_accu=train_new_accu,
                    old_accu=train_old_accu,
                )
                self.curr_iter += 1
                
                loss = loss + loss_ce
                loss_base = loss_base + loss_ce
                
                if self._cfg["use_aux_cls"] and self._num_heads > 1:
                    loss = loss + loss_aux
                    loss_base = loss_base + loss_aux

                if self.enable_pruning:
                    regs = self._network.regularizer()
                    loss_pruning = regs[self._network.active_head]
                    loss_base = loss_base + self.lambda_pruning * loss_pruning
                
                loss_base.backward()
                optimizer_base.step()
                
                for optimizer in optimizers.values():
                    optimizer.zero_grad()
                    optimizer.step()
                
                if self._cfg["novelty_detection"]["enable"]:
                    optimizer_novelty = optimizers['optimizer_novelty']
                    loss = loss + loss_novelty
                    loss_novelty.backward()
                    optimizer_novelty.step()
                    
                    for optimizer in optimizers.values():
                        optimizer.zero_grad()
                        optimizer.step()

                if torch.isnan(loss):
                    raise RuntimeError('Loss has NaN values')

                if self._cfg["postprocessor"]["enable"]:
                    if self._cfg["postprocessor"]["type"].lower() == "wa":
                        for p in self._network.classifier.parameters():
                            p.data.clamp_(0.0)
                        
                        # Not Tested
                        if self._network.use_calibration_head:
                            for p in self._network.calibrated_classifiers.parameters():
                                p.data.clamp_(0.0)

                _loss_ce = _loss_ce + loss_ce
                _loss_aux = _loss_aux + loss_aux
                _loss_novelty = _loss_novelty + loss_novelty
                _loss_pruning = _loss_pruning + loss_pruning
                
            _loss_ce = _loss_ce.item()
            _loss_aux = _loss_aux.item()
            _loss_novelty = _loss_novelty.item()
            _loss_pruning = _loss_pruning.item()
            
            if not self._warmup:
                for scheduler in schedulers.values():
                    scheduler.step()
            
            self._ex.logger.info(
                "Task {}/{}, Epoch {}/{} => Clf loss: {} Aux loss: {} Novelty loss: {}, Pruning loss: {}, Train Accu: {}, Train@5 Acc: {}, old acc:{}".
                format(
                    self._task + 1,
                    self._n_tasks,
                    epoch + 1,
                    self._n_epochs,
                    round(_loss_ce / i, 3),
                    round(_loss_aux / i, 3),
                    round(_loss_novelty / i, 3),
                    round(_loss_pruning / i, 3),
                    round(accu.value()[0], 3),
                    round(accu.value()[1], 3),
                    round(train_old_accu.value()[0], 3),
                ))

            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

        # For the large-scale dataset, we manage the data in the shared memory.
        self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory

        utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "After training")
        utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                   self._increments, "Trainset")
        self._run.info[f"trial{self._trial_i}"][f"task{self._task}_train_accu"] = round(accu.value()[0], 3)

    def _forward_loss(self, inliers, outliers, targets, outhead_classes, inhead_classes, accu=None, new_accu=None, old_accu=None):
        
        inliers = inliers.to(self._device, non_blocking=True)
        outliers = outliers.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)

        inputs = torch.cat((inliers, outliers), 0)
        outputs = self._parallel_network(inputs) 
        loss_novelty = self._compute_novelty_loss(outputs, inliers, outhead_classes, inhead_classes)

        outputs_inlier = outputs
        for key, value in outputs_inlier.items():
            if value is not None:
                outputs_inlier[key] = value[:len(inliers)]

        if accu is not None:
            accu.add(outputs_inlier['logit'], targets)
        
        loss_ce, loss_aux, loss_ce_calibrated = \
            self._compute_cls_loss(targets, outputs_inlier, outhead_classes, inhead_classes)

        loss_novelty = loss_novelty + loss_ce_calibrated
        if torch.isnan(loss_novelty):
            loss_novelty = torch.zeros([1]).cuda()

        return loss_ce, loss_aux, loss_novelty

    def _compute_novelty_loss(self, outputs, inliers, outhead_classes, inhead_classes):
        if self._cfg['novelty_detection']['calibration_head']:
            output_logits = outputs['calibrated_logit']
        else:
            output_logits = outputs['logit']
            
        inlier_logits = output_logits[:len(inliers)]
        outlier_logits = output_logits[len(inliers):]
        
        if self._network.use_multihead_calibration:
            inlier_logits_old = inlier_logits[outhead_classes] # treated as outliers
            inlier_logits = inlier_logits[inhead_classes]
            
            outlier_logits = torch.cat([outlier_logits, inlier_logits_old])
        
        m_in = self._cfg['novelty_detection']['m_in']
        m_out = self._cfg['novelty_detection']['m_out']
        
        novelty_scores = outputs["novelty_scores"]
        if novelty_scores is not None:
            novelty_scores = novelty_scores[:len(inliers)]
            novelty_scores = novelty_scores[inhead_classes]
            if len(novelty_scores) > 0:
                for i in range(novelty_scores.shape[-1]):
                    score = torch.mean(novelty_scores[:, i], dim=-1)
                    self._tensorboard.add_scalar("novelty_{}".format(i), score, self.curr_iter)

            new_means, new_vars = update_exponential_moving_average(
                self._network.novelty_means[self._network.active_head], 
                self._network.novelty_vars[self._network.active_head], 
                novelty_scores[:, self._network.active_head]
            )
            self._network.update_running_novelty_stats(new_means, new_vars)        
        return energy_criterion(inlier_logits, outlier_logits, m_in, m_out)

    def _compute_cls_loss(self, targets, outputs, outhead_classes, inhead_classes):
        loss_ce = F.cross_entropy(outputs['logit'], targets)
        
        loss_ce_calibrated = torch.zeros([1]).cuda()
        if outputs['calibrated_logit'] is not None:
            targets_calib = targets.clone()
            outputs_calib = outputs['calibrated_logit']
            if self._network.use_multihead_calibration:
                targets_calib = targets_calib[inhead_classes]
                targets_calib -= sum(self._inc_dataset.increments[:self._task])
                outputs_calib = outputs_calib[inhead_classes]
            
            loss_ce_calibrated = F.cross_entropy(outputs_calib, targets_calib)
            if torch.isnan(loss_ce_calibrated):
                loss_ce_calibrated = torch.zeros([1]).cuda()

        loss_aux = torch.zeros([1]).cuda()
        if outputs['aux_logit'] is not None:
            targets_aux = targets.clone()
            if self._cfg["aux_n+1"]:
                targets_aux[outhead_classes] = 0
                targets_aux[inhead_classes] -= sum(self._inc_dataset.increments[:self._task]) - 1
            loss_aux = F.cross_entropy(outputs['aux_logit'], targets_aux)
            
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
            self._network.classifier.reset_parameters()
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
            'params': self._network.classifier.parameters(),
            'lr': self._decouple["lr"]
        }]
        # if self._network._cfg['novelty_detection']['calibration_head']:
        #     if self._network.use_multihead_calibration:
        #         params.append({
        #             'params': self._network.calibrated_classifiers[self._network.active_head].parameters(),
        #             'lr': self._cfg["novelty_detection"]["optimizer"]["lr"]
        #         })
        #     else:
        #         params.append({
        #             'params': self._network.calibrated_classifiers.parameters(),
        #             'lr': self._cfg["novelty_detection"]["optimizer"]["lr"]
        #         })
        
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
            
            classes_in_head = self._classes_in_task[self._task]
            
            self._dataloader_out.dataset.offset = np.random.randint(len(self._dataloader_out.dataset))
            generator_out = self._generator_from_loader(self._dataloader_out)
            for inliers, targets in loader:
                outliers, _ = next(generator_out)
                inliers, targets, outliers = inliers.cuda(), targets.cuda(), outliers.cuda()
                inputs = torch.cat((inliers, outliers), 0)
                
                inhead_classes = tensor_is_in(targets, classes_in_head)
                outhead_classes = torch.logical_not(inhead_classes)
                
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
                loss = loss + loss_ce
                
                # if self._cfg["novelty_detection"]["enable"]:
                #     loss = loss + loss_novelty
                
                loss.backward()
                optim.step()
                
                total_ce_loss = total_ce_loss + loss_ce.detach().cpu().numpy() * inliers.size(0)
                total_novelty_loss = total_novelty_loss + loss_novelty.detach().cpu().numpy() * inputs.size(0)
                total_correct = total_correct + (preds == targets).sum()
                total_inlier_count = total_inlier_count + inliers.size(0)
                total_data_count = total_data_count + inputs.size(0)

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
