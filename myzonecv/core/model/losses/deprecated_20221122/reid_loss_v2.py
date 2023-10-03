import torch
import torch.nn as nn

from ....registry import LOSSES, REID_LOSSES
from ...base_module import BaseModule


@REID_LOSSES.register_class('classification_v2')
class ReIDClassificationV2(BaseModule):
    """ Classification loss (BCE) on identification labels """

    def __init__(self, n_dims=None, n_classes=None, enable_temperature=True, label_smoothing=0.2, intra_relax=0.1, inter_relax=0.1, margin=0.01,
                 radial_weight=1., centroid_weight=1., loss_weight=1., loss_name='cls_loss', prefix='', init_cfg=None):
        super().__init__(init_cfg)
        assert n_dims is not None and n_classes is not None
        self.n_dims = n_dims
        self.n_classes = n_classes
        self.enable_temperature = enable_temperature
        self.label_smoothing = label_smoothing
        self.intra_relax = intra_relax
        self.inter_relax = inter_relax
        self.margin = margin
        self.radial_weight = radial_weight
        self.centroid_weight = centroid_weight
        self.loss_weight = loss_weight
        assert loss_name.endswith('_loss')
        self.loss_name = loss_name
        self.prefix = prefix

        self.centroid = nn.parameter.Parameter(torch.randn(n_dims, n_classes))
        self.alpha = nn.parameter.Parameter(torch.tensor(0.)) if enable_temperature else None
        self.cossim = nn.CosineSimilarity(dim=1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x, target):
        """ x: bs x n_dims
            target (dtype=torch.long): bs 
        """
        radial_loss = torch.abs(torch.norm(x, dim=1) - 1).mean()

        if self.alpha is None:
            output = self.cossim(x.unsqueeze(2), self.centroid.unsqueeze(0))  # bs x n_classes
        else:
            output = self.cossim(x.unsqueeze(2), self.centroid.unsqueeze(0)) * torch.exp(self.alpha)  # bs x n_classes

        out_detached = output.detach()
        maxval, maxidx = out_detached.max(dim=1)
        secval, secidx = out_detached.kthvalue(self.n_classes - 1, dim=1)
        correct = maxidx == target
        maxthr = torch.full_like(maxval, 1 - self.intra_relax)
        maxthr = torch.where(correct, torch.minimum(maxthr, secval + self.margin), maxval + self.margin)[:, None]
        output = torch.clamp(output, max=maxthr)
        cls_loss = self.criterion(output, target)

        centroid_sim = self.cossim(self.centroid.T.unsqueeze(2), self.centroid.unsqueeze(0))  # n_classes x n_classes
        centroid_sim = torch.triu(centroid_sim, diagonal=1)
        centroid_loss = torch.clamp(centroid_sim - self.inter_relax, 0).sum()

        loss = (cls_loss + radial_loss * self.radial_weight + centroid_loss * self.centroid_weight) * self.loss_weight

        return {self.loss_name: loss,
                f'{self.prefix}lcls': cls_loss.detach(),
                f'{self.prefix}lrad': radial_loss.detach(),
                f'{self.prefix}lcen': centroid_loss.detach()}


@REID_LOSSES.register_class('metriclearning_v2')
class ReIDMetricLearningV2(BaseModule):
    """ Metric-learning loss including pairwise terms and triplet terms 
    """

    def __init__(self, margin_pos=0.1, margin_neg=0.1, margin_tri=0.1, pos_weight=1., neg_weight=1., tri_weight=1.,
                 loss_weight=1., loss_name='ml_loss', prefix='', init_cfg=None):
        super().__init__(init_cfg)
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.margin_tri = margin_tri
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.tri_weight = tri_weight
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
        dist = torch.cdist(x, x)  # bs x bs
        pos_dist = dist[mask_pos]  # g*m*(m-1)
        neg_dist = dist[mask_neg]  # g*m*(bs-m)
        dist_ratio = neg_dist / torch.clamp(pos_dist[:, None], 1e-10)
        tri_dist = dist_ratio[mask_tri]  # g2*m2*m3

        pos_loss = torch.clamp(pos_dist / 1.3503 - 1 + self.margin_pos, 0).mean()
        neg_loss = torch.clamp(1 - neg_dist / 0.7654 + self.margin_neg, 0).mean()
        tri_loss = torch.clamp(1 - tri_dist + self.margin_tri, 0).mean()
        loss = (pos_loss * self.pos_weight + neg_loss * self.neg_weight + tri_loss * self.tri_weight) * self.loss_weight
        return {self.loss_name: loss,
                f'{self.prefix}lpos': pos_loss.detach(),
                f'{self.prefix}lneg': neg_loss.detach(),
                f'{self.prefix}ltri': tri_loss.detach()}


@LOSSES.register_class('reid_loss_v2')
class ReIDLossV2(BaseModule):
    def __init__(self,
                 reid_head,
                 feature_loss_weight=1.,
                 output_loss_weight=1.,
                 cls_loss={'type': 'classification_v2'},
                 ml_loss={'type': 'metriclearning_v2'},
                 init_cfg=None):
        super().__init__(init_cfg)
        self.cls_loss_cfg = cls_loss
        self.ml_loss_cfg = ml_loss
        assert feature_loss_weight > 0. or output_loss_weight > 0.
        self.feature_loss_weight = feature_loss_weight
        self.output_loss_weight = output_loss_weight

        if feature_loss_weight > 0.:
            for name, mod, abbr in reid_head.get_extractor():
                self._add_loss(name, mod, abbr, feature_loss_weight)

        if output_loss_weight > 0.:
            name, mod, abbr = reid_head.get_aggregator()
            self._add_loss(name, mod, abbr, output_loss_weight)

    def _add_loss(self, head_name, head_mod, head_abbr, weight):
        if self.cls_loss_cfg:
            mod_name, loss_name = f'{head_name}_cls_loss', f'{head_abbr}_cls_loss'
            loss_weight = self.cls_loss_cfg.get('loss_weight', 1.0) * weight
            cls_loss = REID_LOSSES.create(self.cls_loss_cfg, n_dims=head_mod.out_channels, loss_weight=loss_weight, loss_name=loss_name, prefix=f'{head_abbr}_')
            self.add_module(mod_name, cls_loss)

        if self.ml_loss_cfg:
            mod_name, loss_name = f'{head_name}_ml_loss', f'{head_abbr}_ml_loss'
            loss_weight = self.ml_loss_cfg.get('loss_weight', 1.0) * weight
            ml_loss = REID_LOSSES.create(self.ml_loss_cfg, loss_weight=loss_weight, loss_name=loss_name, prefix=f'{head_abbr}_')
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
