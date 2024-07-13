from inclearn.convnet.utils import update_exponential_moving_average
from inclearn.learn.optimizers import MultipleOptimizer
from inclearn.learn.losses import energy_criterion
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter, AverageValueMeter

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


def _compute_loss(cfg, logits, targets, device):

    if cfg["train_head"] == "sigmoid":
        n_classes = cfg["start_class"]
        onehot_targets = utils.to_onehot(targets, n_classes).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
    elif cfg["train_head"] == "softmax":
        loss = F.cross_entropy(logits, targets)
    else:
        raise ValueError()

    return loss


def train(cfg, model, optimizers, device, train_loader, tensorboard=None):
    _loss_ce = 0.0
    _loss_novelty = 0.0
    _loss_pruning = 0.0

    accu = ClassErrorMeter(accuracy=True)
    accu.reset()
    
    model._network.novelty_avgs.data[model._head_to_train] = 0.0
    model._network.novelty_vars.data[model._head_to_train] = 0.0
    model.num_samples = 0.0

    model.train()
    model._dataloader_out.dataset.offset = np.random.randint(len(model._dataloader_out.dataset))
    generator_out = model._generator_from_loader(model._dataloader_out)
    for i, (inliers, targets) in enumerate(train_loader, start=1):
        outliers, _ = next(generator_out)
        
        inliers, targets, outliers = inliers.to(device), targets.to(device), outliers.to(device)
        inputs = torch.cat((inliers, outliers), 0)
        
        outputs = model._parallel_network(inputs)
        logits = outputs['logit']
        logits_inlier = logits[:len(inliers)]
        
        if accu is not None:
            accu.add(logits_inlier.detach(), targets)
        
        loss_base = torch.zeros([1]).to(model._device)

        loss_ce = _compute_loss(cfg, logits_inlier, targets, device)
        loss_base = loss_base + loss_ce
        if torch.isnan(loss_ce):
            raise RuntimeError('CE loss has NaN values')

        if cfg['pruning']['enable']:
            lambda_pruning = cfg['pruning']['lambda']
            regs = model._network.regularizer()
            loss_pruning = regs[model._head_to_train]
            if torch.isnan(loss_pruning):
                raise RuntimeError('Pruning loss has NaN values')
            loss_base = loss_base + lambda_pruning * loss_pruning
        
        loss_base.backward()
        optimizers['optimizer_base'].step()
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        loss_outlier = torch.zeros([1]).to(model.device)
        if cfg["novelty_detection"]["enable"]:
            # Calibration head
            if cfg["novelty_detection"]["calibration_head"]:
                calibrated_logits = outputs['calibrated_logit']
            else:
                calibrated_logits = outputs['logit']
            calibrated_logits_inlier = calibrated_logits[:len(inliers)]
            calibrated_logits_outlier = calibrated_logits[len(inliers):]
            
            m_in = cfg['novelty_detection']['m_in']
            m_out = cfg['novelty_detection']['m_out']
            loss_ce_calibrated = _compute_loss(cfg, calibrated_logits_inlier, targets, device)
            loss_outlier = energy_criterion(calibrated_logits_inlier, calibrated_logits_outlier, m_in, m_out)
            
            loss_novelty = loss_ce_calibrated + loss_outlier
            loss_novelty.backward()
            optimizers['optimizer_novelty'].step()
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            
            novelty_scores = outputs["novelty_scores"]
            if novelty_scores is not None:
                novelty_scores = novelty_scores[:len(inliers)]
                for head_idx in range(novelty_scores.shape[-1]):
                    score = torch.mean(novelty_scores[:, head_idx], dim=-1)
                    tensorboard.add_scalar("novelty_{}".format(head_idx), score, model.curr_iter)
                model.curr_iter += 1
                
                model._network.novelty_avgs.data[model._head_to_train], \
                model._network.novelty_vars.data[model._head_to_train] = update_exponential_moving_average(
                    model._network.novelty_avgs[model._head_to_train], 
                    model._network.novelty_vars[model._head_to_train], 
                    novelty_scores[:, model._head_to_train], 
                )

        _loss_ce += loss_ce
        _loss_novelty += loss_novelty
        _loss_pruning += loss_pruning

    return (
        round(_loss_ce.item() / i, 3),
        round(_loss_novelty.item() / i, 3),
        round(_loss_pruning.item() / i, 3),
        round(accu.value()[0], 3),
    )


def test(cfg, model, device, test_loader):
    _loss_ce = 0.0
    accu = ClassErrorMeter(accuracy=True)
    accu.reset()

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader, start=1):
            # assert torch.isnan(inputs).sum().item() == 0
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model._parallel_network(inputs)['logit']
            if accu is not None:
                accu.add(logits.detach(), targets)
            loss_ce = _compute_loss(cfg, logits, targets, device)
            
            if torch.isnan(loss_ce):
                raise RuntimeError('Loss has NaN values')

            _loss_ce += loss_ce
    return round(_loss_ce.item() / i, 3), round(accu.value()[0], 3)

def pretrain(cfg, ex, model, device, train_loader, test_loader, model_path, tensorboard=None):
    ex.logger.info(f"nb Train {len(train_loader.dataset)} Eval {len(test_loader.dataset)}")

    optimizers = {}
    schedulers = {}
    
    optimizer_base = factory.get_optimizer(
        model._network.standard_parameters(),
        cfg["optimizer"], 
        cfg["pretrain"]["lr"], 
        weight_decay=cfg["pretrain"]["weight_decay"]
    )
    scheduler_base = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_base,
        cfg["pretrain"]["scheduling"],
        gamma=cfg["pretrain"]["lr_decay"]
    )
    optimizers['optimizer_base'] = optimizer_base
    schedulers['scheduler_base'] = scheduler_base

    if cfg["novelty_detection"]["enable"]:
        optimizer_novelty = factory.get_optimizer(
            model._network.calibration_parameters(),
            cfg["optimizer"], 
            cfg["novelty_detection"]["optimizer"]["lr"], 
            weight_decay=cfg["pretrain"]["weight_decay"]
        )
        scheduler_novelty = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_novelty,
            cfg["pretrain"]["scheduling"],
            gamma=cfg["pretrain"]["lr_decay"]
        )
        optimizers['optimizer_novelty'] = optimizer_novelty
        schedulers['scheduler_novelty'] = scheduler_novelty
        
    test_loss, test_acc = float("nan"), float("nan")
    for e in range(cfg["pretrain"]["epochs"]):
        train_loss_ce, train_loss_novelty, train_loss_pruning, train_acc = \
            train(cfg, model, optimizers, device, train_loader, tensorboard)
        if e % 5 == 0:
            test_loss, test_acc = test(cfg, model, device, test_loader)
            ex.logger.info(
                "Pretrain Class {}, Epoch {}/{} => Clf Train loss CE: {}, Novelty: {}, Pruning: {}, Accu {} | Eval loss: {}, Accu {}".format(
                    cfg["start_class"], 
                    e + 1, 
                    cfg["pretrain"]["epochs"], 
                    train_loss_ce, 
                    train_loss_novelty, 
                    train_loss_pruning,
                    train_acc, 
                    test_loss, 
                    test_acc))
        else:
            ex.logger.info("Pretrain Class {}, Epoch {}/{} => Clf Train loss CE: {}, Novelty: {}, Pruning: {}, Accu {} ".format(
                cfg["start_class"], 
                e + 1, 
                cfg["pretrain"]["epochs"], 
                train_loss_ce, 
                train_loss_novelty, 
                train_loss_pruning,
                train_acc))
        
        for scheduler in schedulers.values():
            scheduler.step()
        
        if e % 50 == 0:
            if hasattr(model._network, "module"):
                torch.save(model._network.module.state_dict(), model_path)
            else:
                torch.save(model._network.state_dict(), model_path)
