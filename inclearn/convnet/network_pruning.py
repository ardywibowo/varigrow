import copy
from inclearn.convnet.pruning_layers import PruningConv2d

import torch
import torch.nn.functional as F
from inclearn.convnet.classifier import CosineClassifier
from inclearn.convnet.imbalance import WA, BiC
from inclearn.tools import factory
from torch import nn
from torch.nn import Parameter


class BasicNet(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet, self).__init__()
        self._cfg = cfg
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']
        
        self.use_calibration_head = cfg['novelty_detection']['calibration_head']
        self.use_multihead_calibration = cfg['novelty_detection']['multihead']
        self.use_calibration_backbone = cfg['novelty_detection']['calibration_backbone']
        self.temp = cfg['novelty_detection']['T']
        
        self.active_head = 0

        self.base_novelty_mean = Parameter(torch.tensor([0.0]), requires_grad=False)
        self.base_novelty_var = Parameter(torch.tensor([0.0]), requires_grad=False)

        self.novelty_means = Parameter(torch.tensor([]), requires_grad=False)
        self.novelty_vars = Parameter(torch.tensor([]), requires_grad=False)

        if self.der:
            print("Enable dynamical representation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            
            if self.use_calibration_backbone:
                self.convnets_calib = nn.ModuleList()
                self.convnets_calib.append(
                    factory.get_convnet(convnet_type,
                                        nf=nf,
                                        dataset=dataset,
                                        start_class=self.start_class,
                                        remove_last_relu=self.remove_last_relu))
            
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        
        if self.use_multihead_calibration:
            self.calibrated_classifiers = nn.ModuleList()
        else:
            self.calibrated_classifiers = None
        
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)
        
        calibrated_logits = None 
        if self.calibrated_classifiers is not None:
            if self.use_calibration_backbone:
                calibrated_features = [convnet(x) for convnet in self.convnets_calib]
                calibrated_features = torch.cat(calibrated_features, dim=1)
            else:
                calibrated_features = features.detach()
            
            if self.use_multihead_calibration:
                calibrated_logits = self.calibrated_classifiers[self.active_head](
                    calibrated_features[:, self.out_dim * (len(self.convnets) - 1) : ]
                )
                novelty_scores = []
                for i, calibrated_classifier in enumerate(self.calibrated_classifiers):
                    calib_head = calibrated_classifier(
                        calibrated_features[:, self.out_dim * i : self.out_dim * (i+1)]
                    )
                    novelty_score = -self.temp * torch.logsumexp(calib_head / self.temp, dim=1)
                    novelty_scores.append(novelty_score)
                novelty_scores = torch.stack(novelty_scores, dim=-1)
            else:
                calibrated_logits = self.calibrated_classifiers(calibrated_features)
                novelty_scores = -self.temp * torch.logsumexp(calibrated_logits / self.temp, dim=1)
                novelty_scores = novelty_scores.unsqueeze(-1)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) \
            if features.shape[1] > self.out_dim else None
            
        return {
            'feature': features, 
            'logit': logits, 
            'aux_logit': aux_logits,
            'calibrated_logit': calibrated_logits,
            'novelty_scores': novelty_scores,
            # 'calibrated_feature': calibrated_features,
        }

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(
                self.convnet_type,
                nf=self.nf,
                dataset=self.dataset,
                start_class=self.start_class,
                remove_last_relu=self.remove_last_relu
            ).to(self.device)
            
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)
            
            if self.use_calibration_backbone:
                new_clf = factory.get_convnet(
                    self.convnet_type,
                    nf=self.nf,
                    dataset=self.dataset,
                    start_class=self.start_class,
                    remove_last_relu=self.remove_last_relu
                ).to(self.device)
                new_clf.load_state_dict(self.convnets_calib[-1].state_dict())
                self.convnets_calib.append(new_clf)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            
        if self.calibrated_classifiers is not None and not self.use_multihead_calibration:
            calibrated_weight = copy.deepcopy(self.calibrated_classifiers.weight.data)

        fc = self._gen_classifier(
            self.out_dim * len(self.convnets), 
            self.n_classes + n_classes
        )
        
        # fc_calibrated = self._gen_classifier(
        #     self.out_dim * len(self.convnets), 
        #     n_classes
        # )
        
        if self.use_multihead_calibration:
            fc_calibrated = self._gen_classifier(self.out_dim, n_classes)
        else:
            fc_calibrated = self._gen_classifier(
                self.out_dim * len(self.convnets), 
                self.n_classes + n_classes
            )
            
        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
            if self.use_multihead_calibration:
                fc_calibrated.weight.data[
                    :self.n_classes, :self.out_dim * (len(self.convnets) - 1)
                ] = calibrated_weight
        
        del self.classifier
        if not self.use_multihead_calibration:
            del self.calibrated_classifiers
        
        self.classifier = fc
        if self.use_calibration_head:
            if self.use_multihead_calibration:
                self.calibrated_classifiers.append(fc_calibrated)
            else:
                self.calibrated_classifiers = fc_calibrated

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)
        
        if self.calibrated_classifiers is not None and not self.use_multihead_calibration:
            calibrated_weight = copy.deepcopy(self.calibrated_classifiers.weight.data)
            if self.use_bias:
                calibrated_bias = copy.deepcopy(self.calibrated_classifiers.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)
        # calibrated_classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)
        # calibrated_classifier = self._gen_classifier(self.features_dim, n_classes)
        
        if self.use_multihead_calibration:
            calibrated_classifier = self._gen_classifier(self.out_dim, n_classes)
        else:
            calibrated_classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)
            

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias
        
        if self.calibrated_classifiers is not None \
            and self.reuse_oldfc and \
            not self.use_multihead_calibration:
            
            calibrated_classifier.weight.data[:self.n_classes] = calibrated_weight
            if self.use_bias:
                calibrated_classifier.bias.data[:self.n_classes] = calibrated_bias

        del self.classifier
        self.classifier = classifier
        
        if self.use_calibration_head:
            
            if self.use_multihead_calibration:
                self.calibrated_classifiers.append(calibrated_classifier)
            else:
                del self.calibrated_classifiers
                self.calibrated_classifiers = calibrated_classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier

    def calibration_parameters(self):
        if not self.use_calibration_head:
            return None
        elif not self.use_calibration_backbone:
            return self.calibrated_classifiers.parameters()
        else:
            params = list(self.convnets_calib.parameters()) + \
                list(self.calibrated_classifiers.parameters())
            
            return params

    def standard_parameters(self):
        params = list(self.convnets.parameters()) + \
            list(self.classifier.parameters())
        
        if self.aux_classifier is not None:
            params += list(self.aux_classifier.parameters())
        
        return params

    def update_running_novelty_stats(self, novelty_avg, novelty_var):
        self.novelty_means.data[self._head_to_train] = novelty_avg.clone().detach().to(self.device)
        self.novelty_vars.data[self._head_to_train] = novelty_var.clone().detach().to(self.device)
        
    def update_base_novelty_stats(self, novelty_avg, novelty_var):
        self.base_novelty_mean.data = novelty_avg.clone().detach().to(self.device)
        self.base_novelty_var.data = 4.0 * novelty_var.clone().detach().to(self.device)
        
    def grow_novelty_stats(self):
        self.novelty_means.data = torch.cat([self.novelty_means.data, self.base_novelty_mean.data])
        self.novelty_vars.data = torch.cat([self.novelty_vars.data, self.base_novelty_var.data])

    def regularizer(self):
        regs = []
        for net in self.convnets:
            num_active = torch.zeros([1]).to(self.device)
            num_total = torch.zeros([1]).to(self.device)

            active_prev, total_prev = None, None
            for m in net.modules():
                if isinstance(m, PruningConv2d):
                    if active_prev is None:
                        active_prev, total_prev = m.reg()
                    else:
                        active_curr, total_curr = m.reg()

                        num_active = num_active + (active_curr * active_prev)
                        num_total = num_total + (total_curr * total_prev)

                        active_prev, total_prev = active_curr, total_curr
            
            reg_curr = num_active / num_total
            regs.append(reg_curr)
        
        return torch.cat(regs)