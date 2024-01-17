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


def train(cfg, model, optimizer, device, train_loader):
    _loss_ce = 0.0
    _loss_novelty = 0.0
    accu = ClassErrorMeter(accuracy=True)
    accu.reset()

    model.train()

    model._dataloader_out.dataset.offset = np.random.randint(len(model._dataloader_out.dataset))
    generator_out = model._generator_from_loader(model._dataloader_out)
    for i, (inliers, targets) in enumerate(train_loader, start=1):
        outliers, _ = next(generator_out)
        
        optimizer.zero_grad()
        inliers, targets, outliers = inliers.to(device), targets.to(device), outliers.to(device)
        inputs = torch.cat((inliers, outliers), 0)
        
        outputs = model._parallel_network(inputs)
        logits = outputs['logit']
        logits_inlier = logits[:len(inliers)]
        
        if accu is not None:
            accu.add(logits_inlier.detach(), targets)
            
        loss = torch.zeros([1]).cuda()
        loss_ce = _compute_loss(cfg, logits_inlier, targets, device)
        loss += loss_ce
        
        loss_novelty = torch.zeros([1]).cuda()
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
            loss_novelty = energy_criterion(calibrated_logits_inlier, calibrated_logits_outlier, m_in, m_out)
            
            loss += loss_ce_calibrated + loss_novelty
        
        if torch.isnan(loss_ce):
            raise RuntimeError('Loss has NaN values')

        loss.backward()
        optimizer.step()
        _loss_ce += loss_ce
        _loss_novelty += loss_novelty

    return (
        round(_loss_ce.item() / i, 3),
        round(_loss_novelty.item() / i, 3),
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

            _loss_ce = _loss_ce + loss_ce
    return round(_loss_ce.item() / i, 3), round(accu.value()[0], 3)


def pretrain(cfg, ex, model, device, train_loader, test_loader, model_path):
    ex.logger.info(f"nb Train {len(train_loader.dataset)} Eval {len(test_loader.dataset)}")
    
    params = [
        {'params': model._network.standard_parameters(),
         'lr': cfg["pretrain"]["lr"]},
    ]
    
    if cfg["novelty_detection"]["enable"]:
        params.append({
            'params': model._network.calibration_parameters(),
            'lr': cfg["novelty_detection"]["optimizer"]["lr"]
        })
    
    optimizer = factory.get_optimizer(
        params,
        cfg["optimizer"], 
        cfg["pretrain"]["lr"], 
        weight_decay=cfg["pretrain"]["weight_decay"]
    )

    # optimizer = factory.get_optimizer(model._network.parameters(),
    #                                   cfg["optimizer"], 
    #                                   cfg["pretrain"]["lr"], 
    #                                   weight_decay=cfg["pretrain"]["weight_decay"])

    # optimizer = factory.get_optimizer(list(model._network.classifier.parameters()) + 
    #                                   list(model._network.convnets.parameters()), 
    #                                   cfg["optimizer"], 
    #                                   cfg["pretrain"]["lr"], 
    #                                   weight_decay=cfg["pretrain"]["weight_decay"])
    
    # optimizer_calibrated = factory.get_optimizer(list(model._network.convnets_calib.parameters()) +
    #                                   list(model._network.calibrated_classifier.parameters()), 
    #                                   cfg["novelty_detection"]["optimizer"]["method"], 
    #                                   cfg["novelty_detection"]["optimizer"]["lr"], 
    #                                   weight_decay=cfg["novelty_detection"]["optimizer"]["weight_decay"])
    
    # optimizer = MultipleOptimizer(optimizer, optimizer_calibrated)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     cfg["pretrain"]["scheduling"],
                                                     gamma=cfg["pretrain"]["lr_decay"])
    test_loss, test_acc = float("nan"), float("nan")
    for e in range(cfg["pretrain"]["epochs"]):
        train_loss_ce, train_loss_novelty, train_acc = train(cfg, model, optimizer, device, train_loader)
        if e % 5 == 0:
            test_loss, test_acc = test(cfg, model, device, test_loader)
            ex.logger.info(
                "Pretrain Class {}, Epoch {}/{} => Clf Train loss CE: {}, Novelty: {}, Accu {} | Eval loss: {}, Accu {}".format(
                    cfg["start_class"], 
                    e + 1, 
                    cfg["pretrain"]["epochs"], 
                    train_loss_ce, 
                    train_loss_novelty, 
                    train_acc, 
                    test_loss, 
                    test_acc))
        else:
            ex.logger.info("Pretrain Class {}, Epoch {}/{} => Clf Train loss CE: {}, Novelty: {}, Accu {} ".format(
                cfg["start_class"], 
                e + 1, 
                cfg["pretrain"]["epochs"], 
                train_loss_ce, 
                train_loss_novelty, 
                train_acc))
        scheduler.step()
        if e % 50 == 0:
            if hasattr(model._network, "module"):
                torch.save(model._network.module.state_dict(), model_path)
            else:
                torch.save(model._network.state_dict(), model_path)
