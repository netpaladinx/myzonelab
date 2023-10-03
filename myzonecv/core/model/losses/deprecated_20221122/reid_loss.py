import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import LOSSES, REID_LOSSES
from ...base_module import BaseModule


@REID_LOSSES.register_class('classification')
class ReIDClassification(BaseModule):
    """ Classification loss (BCE) on identification labels """

    def __init__(self, n_dims=None, n_classes=None, radial_weight=1., radial_min=0.6, radial_max=1., radial_p=2, enable_temperature=True,
                 label_smoothing=0., loss_weight=1., loss_name='cls_loss', prefix='', init_cfg=None):
        super().__init__(init_cfg)
        assert n_dims is not None and n_classes is not None
        self.n_dims = n_dims
        self.n_classes = n_classes
        self.radial_weight = radial_weight
        self.radial_min = radial_min
        self.radial_max = radial_max
        self.radial_p = radial_p
        self.enable_temperature = enable_temperature
        self.label_smoothing = label_smoothing
        self.loss_weight = loss_weight
        assert loss_name.endswith('_loss')
        self.loss_name = loss_name
        self.prefix = prefix

        self.weight = nn.parameter.Parameter(torch.randn(n_dims, n_classes))
        self.alpha = nn.parameter.Parameter(torch.tensor(0.)) if enable_temperature else None
        self.cossim = nn.CosineSimilarity()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x, target):
        """ x: bs x n_dims
            target (dtype=torch.long): bs 
        """
        if self.alpha is None:
            output = self.cossim(x.unsqueeze(2), self.weight.unsqueeze(0))  # bs x n_classes
        else:
            output = self.cossim(x.unsqueeze(2), self.weight.unsqueeze(0)) * torch.exp(self.alpha)  # bs x n_classes
        cls_loss = self.criterion(output, target)
        radial_length = torch.norm(x, p=self.radial_p, dim=1)  # bs
        radial_loss = torch.clamp(radial_length - self.radial_max, 0).mean() + torch.clamp(self.radial_min - radial_length, 0).mean()
        loss = (cls_loss + self.radial_weight * radial_loss) * self.loss_weight
        return {self.loss_name: loss,
                f'{self.prefix}lcls': cls_loss.detach(),
                f'{self.prefix}lrad': radial_loss.detach()}


@REID_LOSSES.register_class('metriclearning')
class ReIDMetricLearning(BaseModule):
    """ Metric-learning loss including pairwise terms and triplet terms 

        For any v such that ndim(v) = 128, 0.6 <= ||v||_2 <= 1:
        1) if p=1, then margin_pos=1.0, margin_neg=8., margin_tri=7.
        2) if p=2, then margin_pos=0.4, margin_neg=0.8, margin_tri=0.4
    """

    def __init__(self, margin_pos=1., margin_neg=8., margin_tri=7., pos_weight=1., neg_weight=1., tri_weight=0.2,
                 p=1, hard_mining_mode=None,
                 loss_weight=1., loss_name='ml_loss', prefix='', init_cfg=None):
        super().__init__(init_cfg)
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.margin_tri = margin_tri
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.tri_weight = tri_weight
        self.p = p
        assert hard_mining_mode is None or hard_mining_mode in ('sample-level', 'batch-level')
        self.hard_mining_mode = hard_mining_mode
        self.loss_weight = loss_weight
        assert loss_name.endswith('_loss')
        self.loss_name = loss_name
        self.prefix = prefix

    def forward(self, x, mask_pos, mask_neg, mask_tri):
        """ x: bs x n_dims (bs = g*m, e.g. 64 = 8*8) (g: groups, m: members)
            mask_pos: bs x bs (#Trues = g*m*(m-1), repeated twice)
            mask_neg: bs x bs (#Trues = g*m*(bs-m), repeated twice)
            mask_tri: (g*m*(m-1)) x g*m*(bs-m) (#Trues = g2*m2*m3, repeated twice) where g2 = g*m, m2 = m-1, m3 = bs-m
        """
        dist = torch.cdist(x, x, p=self.p)  # bs x bs
        pos_dist = dist[mask_pos]  # g*m*(m-1)
        neg_dist = dist[mask_neg]  # g*m*(bs-m)
        dist_diff = pos_dist[:, None] - neg_dist
        tri_dist = dist_diff[mask_tri]  # g2*m2*m3

        if self.hard_mining_mode == 'sample-level':
            pos_dist = F.adaptive_max_pool1d((dist * mask_pos).unsqueeze(0), 1)
            M = dist.detach().max()
            neg_dist = M - F.adaptive_max_pool1d(((M - dist) * mask_neg).unsqueeze(0), 1)
            tri_dist = F.adaptive_max_pool1d((dist_diff * mask_tri).unsqueeze(0), 1)
        elif self.hard_mining_mode == 'batch-level':
            pos_dist = F.adaptive_max_pool2d((dist * mask_pos).unsqueeze(0), 1)
            M = dist.detach().max()
            neg_dist = M - F.adaptive_max_pool2d(((M - dist) * mask_neg).unsqueeze(0), 1)
            tri_dist = F.adaptive_max_pool2d((dist_diff * mask_tri).unsqueeze(0), 1)

        pos_loss = torch.clamp(pos_dist - self.margin_pos, 0).mean()
        neg_loss = torch.clamp(self.margin_neg - neg_dist, 0).mean()
        tri_loss = torch.clamp(tri_dist + self.margin_tri, 0).mean()
        loss = (pos_loss * self.pos_weight + neg_loss * self.neg_weight + tri_loss * self.tri_weight) * self.loss_weight
        return {self.loss_name: loss,
                f'{self.prefix}lpos': pos_loss.detach(),
                f'{self.prefix}lneg': neg_loss.detach(),
                f'{self.prefix}ltri': tri_loss.detach()}


@LOSSES.register_class('reid_loss')
class ReIDLoss(BaseModule):
    def __init__(self,
                 reid_head,
                 apply_final_loss=True,
                 cls_loss={'type': 'classification'},
                 ml_loss={'type': 'metriclearning'},
                 init_cfg=None):
        super().__init__(init_cfg)
        self.apply_final_loss = apply_final_loss
        self.cls_loss_cfg = cls_loss
        self.ml_loss_cfg = ml_loss

        for name, mod, abbr in reid_head.get_global_extractor():
            n_feats = mod.n_features
            if n_feats == 1:
                self._add_loss(name, mod, abbr)
            else:
                for i in range(n_feats):
                    self._add_loss(f'{name}_{i}', mod, f'{abbr}{i}')

        for name, mod, abbr in reid_head.get_local_extractor():
            n_feats = mod.n_features
            if n_feats == 1:
                self._add_loss(name, mod, abbr)
            else:
                for i in range(n_feats):
                    self._add_loss(f'{name}_{i}', mod, f'{abbr}{i}')

        if apply_final_loss:
            name, mod, abbr = reid_head.get_final_aggregator()
            if mod.out_channels is not None:
                self._add_loss(name, mod, abbr)

    def _add_loss(self, head_name, head_mod, head_abbr):
        if self.cls_loss_cfg:
            mod_name, loss_name = f'{head_name}_cls_loss', f'{head_abbr}_cls_loss'
            cls_loss = REID_LOSSES.create(self.cls_loss_cfg, n_dims=head_mod.out_channels, loss_name=loss_name, prefix=f'{head_abbr}_')
            self.add_module(mod_name, cls_loss)

        if self.ml_loss_cfg:
            mod_name, loss_name = f'{head_name}_ml_loss', f'{head_abbr}_ml_loss'
            ml_loss = REID_LOSSES.create(self.ml_loss_cfg, loss_name=loss_name, prefix=f'{head_abbr}_')
            self.add_module(mod_name, ml_loss)

    def forward(self, x_dict, cls_target, mask_pos, mask_neg, mask_tri):
        """ x_dict:
                'global_average_extractor': B x D
                'final_concat_aggregator': B x D
            cls_target: B (dtype: torch.int64)
            mask_pos: B x B
            mask_neg: B x B
            mask_tri: #Trues-in-mask_pos x #Trues-in-mask_neg
        """
        loss_res = {}

        for name, x in x_dict.items():
            cls_loss_name = f'{name}_cls_loss'
            if hasattr(self, cls_loss_name):
                cls_loss = getattr(self, cls_loss_name)
                res = cls_loss(x, cls_target)
                loss_res.update(res)

            ml_loss_name = f'{name}_ml_loss'
            if hasattr(self, ml_loss_name):
                ml_loss = getattr(self, ml_loss_name)
                res = ml_loss(x, mask_pos, mask_neg, mask_tri)
                loss_res.update(res)

        return loss_res
