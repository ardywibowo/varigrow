from typing import Collection, Optional, Union
from inclearn.tools import factory
import numpy as np
import torch
import torch.nn.functional as F
from ignite.utils import to_onehot
from inclearn.learn.losses import energy_criterion
from inclearn.tools.metrics import AverageValueMeter, ClassErrorMeter
from torch import nn
from torch.optim import SGD


def finetune_last_layer(
    logger,
    network,
    loader,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    loss_type="ce",
    temperature=5.0,
    test_loader=None,
    optimizer='adam'
):
    network.eval()
    #if hasattr(network.module, "convnets"):
    #    for net in network.module.convnets:
    #        net.eval()
    #else:
    #    network.module.convnet.eval()
    
    params = [{'classifier': network.module.classifier.parameters()}]
    if network.module._cfg['novelty_detection']['calibration_head']:
        params.append({'calibrated_classifier': network.module.calibrated_classifier.parameters()})
    
    optim = factory.get_optimizer(
        network.module.classifier.parameters(),
        optimizer, 
        lr, 
        weight_decay)
    
    # optim = SGD(network.module.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Begin finetuning last layer")

    for i in range(nepoch):
        total_ce_loss = 0.0
        total_novelty_loss = 0.0
        total_correct = 0.0
        total_inlier_count = 0
        total_data_count = 0
        # print(f"dataset loader length {len(loader.dataset)}")
        
        network.module._dataloader_out.dataset.offset = np.random.randint(len(network.module._dataloader_out.dataset))
        generator_out = network.module._generator_from_loader(network.module._dataloader_out)
        for inliers, targets in loader:
            outliers, _ = next(generator_out)
            inliers, targets, outliers = inliers.cuda(), targets.cuda(), outliers.cuda()
            inputs = torch.cat((inliers, outliers), 0)
            
            if loss_type == "bce":
                targets = to_onehot(targets, n_class)
            outputs = network(inputs)
            logits = outputs['logit']
            logits_inlier = logits[:len(inliers)]
            
            _, preds = outputs.max(1)
            optim.zero_grad()
            loss_ce = criterion(logits_inlier / temperature, targets)
            
            if network.module._cfg['novelty_detection']['calibration_head']:
                calibration_logits = outputs['calibrated_logit']
            else:
                calibration_logits = outputs['logit']
            
            calibration_inlier = calibration_logits[:len(inliers)]
            calibration_outlier = calibration_logits[len(inliers):]
            
            m_in = network.module._cfg['novelty_detection']['m_in']
            m_out = network.module._cfg['novelty_detection']['m_out']
            loss_novelty = energy_criterion(calibration_inlier, calibration_outlier, m_in, m_out)
            
            loss = torch.zeros([1]).cuda()
            loss += loss_ce
            
            if network.module._cfg["novelty_detection"]["enable"]:
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
                    outputs = network(inliers.cuda())['logit']
                    _, preds = outputs.max(1)
                    test_correct += (preds.cpu() == targets).sum().item()
                    test_count += inliers.size(0)

        scheduler.step()
        if test_loader is not None:
            logger.info(
                "Epoch %d finetuning CE loss %.3f, Novelty loss %.3f, acc %.3f, Eval %.3f" %
                (i, 
                 total_ce_loss.item() / total_inlier_count, 
                 total_novelty_loss.item() / total_data_count, 
                 total_correct.item() / total_inlier_count, 
                 test_correct / test_count))
        else:
            logger.info("Epoch %d finetuning CE loss %.3f, Novelty loss %.3f, acc %.3f" %
                        (i, 
                         total_ce_loss.item() / total_inlier_count, 
                         total_novelty_loss.item() / total_data_count,
                         total_correct.item() / total_inlier_count))
    return network


def extract_features(model, loader):
    targets, features = [], []
    model.eval()
    with torch.no_grad():
        for _inputs, _targets in loader:
            _inputs = _inputs.cuda()
            _targets = _targets.numpy()
            _features = model(_inputs)['feature'].detach().cpu().numpy()
            features.append(_features)
            targets.append(_targets)

    return np.concatenate(features), np.concatenate(targets)


def calc_class_mean(network, loader, class_idx, metric):
    EPSILON = 1e-8
    features, targets = extract_features(network, loader)
    # norm_feats = features/(np.linalg.norm(features, axis=1)[:,np.newaxis]+EPSILON)
    # examplar_mean = norm_feats.mean(axis=0)
    examplar_mean = features.mean(axis=0)
    if metric == "cosine" or metric == "weight":
        examplar_mean /= (np.linalg.norm(examplar_mean) + EPSILON)
    return examplar_mean


def update_classes_mean(network, inc_dataset, n_classes, task_size, share_memory=None, metric="cosine", EPSILON=1e-8):
    loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                     inc_dataset.targets_inc,
                                     shuffle=False,
                                     share_memory=share_memory,
                                     mode="test")
    class_means = np.zeros((n_classes, network.module.features_dim))
    count = np.zeros(n_classes)
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            feat = network(x.cuda())['feature']
            for lbl in torch.unique(y):
                class_means[lbl] += feat[y == lbl].sum(0).cpu().numpy()
                count[lbl] += feat[y == lbl].shape[0]
        for i in range(n_classes):
            class_means[i] /= count[i]
            if metric == "cosine" or metric == "weight":
                class_means[i] /= (np.linalg.norm(class_means) + EPSILON)
    return class_means

def update_exponential_moving_average(running_mean, running_var, curr_samples, momentum=0.1):
    with torch.no_grad():
        mean = curr_samples.mean()
        var = curr_samples.var()
        n = curr_samples.shape[0]
        
        running_mean = momentum * mean + (1 - momentum) * running_mean
        running_var = momentum * var * n / (n - 1) + (1 - momentum) * running_var
    return running_mean, running_var
    
def tensor_is_in(
    query_tensor: torch.LongTensor,
    test_tensor: Union[Collection[int], torch.LongTensor],
    max_id: Optional[int] = None,
    invert: bool = False,
) -> torch.BoolTensor:
    """
    Return a boolean mask with ``Q[i]`` in T.
    The method guarantees memory complexity of ``max(size(Q), size(T))`` and is thus, memory-wise, superior to naive
    broadcasting.
    :param query_tensor: shape: S
        The query Q.
    :param test_tensor:
        The test set T.
    :param max_id:
        A maximum ID. If not given, will be inferred.
    :param invert:
        Whether to invert the result.
    :return: shape: S
        A boolean mask.
    """
    # normalize input
    if not isinstance(test_tensor, torch.Tensor):
        test_tensor = torch.as_tensor(data=list(test_tensor), dtype=torch.long).to(query_tensor.device)
    if max_id is None:
        max_id = max(query_tensor.max(), test_tensor.max()) + 1
    mask = torch.zeros(max_id, dtype=torch.bool).to(query_tensor.device)
    mask[test_tensor] = True
    if invert:
        mask = ~mask
    return mask[query_tensor.view(-1)].view(*query_tensor.shape)