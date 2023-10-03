import torch
import torch.nn as nn

from ...registry import LOSSES, REID_LOSSES
from ..base_module import BaseModule


@REID_LOSSES.register_class('classification')
class ReIDClassification(BaseModule):
    """ Classification loss (BCE) on identification labels """

    def __init__(self, n_dims=None, n_classes=None, enable_temperature=True, label_smoothing=0., intra_relax=0.3, inter_relax=0., margin=0.1,
                 smoothing_mask=0., radial_weight=1., centroid_weight=1., loss_weight=1., loss_name='cls_loss', prefix='', init_cfg=None):
        super().__init__(init_cfg)
        assert n_dims is not None and n_classes is not None
        self.n_dims = n_dims
        self.n_classes = n_classes
        self.enable_temperature = enable_temperature
        self.label_smoothing = label_smoothing
        self.intra_relax = intra_relax
        self.inter_relax = inter_relax
        self.margin = margin
        self.smoothing_mask_rel = smoothing_mask[0] if isinstance(smoothing_mask, (list, tuple)) else smoothing_mask
        self.smoothing_mask_abs = smoothing_mask[1] if isinstance(smoothing_mask, (list, tuple)) else smoothing_mask
        self.smoothing_mask_base = smoothing_mask[2] if isinstance(smoothing_mask, (list, tuple)) else smoothing_mask
        self.radial_weight = radial_weight
        self.centroid_weight = centroid_weight
        self.loss_weight = loss_weight
        assert loss_name.endswith('_loss')
        self.loss_name = loss_name
        self.prefix = prefix

        self.centroid = nn.parameter.Parameter(torch.randn(n_dims, n_classes))
        self.alpha = nn.parameter.Parameter(torch.tensor(0.)) if enable_temperature else None
        self.cossim = nn.CosineSimilarity(dim=1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

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
        correct = maxidx == target  # bs

        mask_abs = torch.where(correct, maxval < 1 - self.intra_relax, True)
        mask_rel = torch.where(correct, maxval < secval + self.margin, True)

        cls_loss = torch.mean(self.criterion(output, target) * (
            mask_abs * mask_rel + mask_abs * self.smoothing_mask_rel + self.smoothing_mask_abs * mask_rel + self.smoothing_mask_base) / (
                1 + self.smoothing_mask_rel + self.smoothing_mask_abs + self.smoothing_mask_base))

        centroid_sim = self.cossim(self.centroid.T.unsqueeze(2), self.centroid.unsqueeze(0))  # n_classes x n_classes
        centroid_sim = torch.triu(centroid_sim, diagonal=1)
        centroid_loss = torch.clamp(centroid_sim - self.inter_relax, 0).mean()

        loss = (cls_loss + radial_loss * self.radial_weight + centroid_loss * self.centroid_weight) * self.loss_weight

        return {self.loss_name: loss,
                f'{self.prefix}lcls': cls_loss.detach(),
                f'{self.prefix}lrad': radial_loss.detach(),
                f'{self.prefix}lcen': centroid_loss.detach()}


@REID_LOSSES.register_class('metriclearning')
class ReIDMetricLearning(BaseModule):
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


@LOSSES.register_class('reid_loss')
class ReIDLoss(BaseModule):
    def __init__(self,
                 reid_head,
                 cls_loss={'type': 'classification'},
                 ml_loss={'type': 'metriclearning'},
                 loss_weight=1.0,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.cls_loss_cfg = cls_loss
        self.ml_loss_cfg = ml_loss
        self.loss_weight = loss_weight

        loss_weight = self.cls_loss_cfg.get('loss_weight', 1.0) * loss_weight
        self.cls_loss = REID_LOSSES.create(self.cls_loss_cfg, n_dims=reid_head.out_channels, loss_weight=loss_weight)

        loss_weight = self.ml_loss_cfg.get('loss_weight', 1.0) * loss_weight
        self.ml_loss = REID_LOSSES.create(self.ml_loss_cfg, loss_weight=loss_weight)

    def forward(self, x_dict, cls_target, mask_pos, mask_neg, mask_tri):
        """ x_dict:
                'output': B x D
            cls_target: B (dtype: torch.int64)
            mask_pos: B x B
            mask_neg: B x B
            mask_tri: #Trues-in-mask_pos x #Trues-in-mask_neg
        """
        loss_res = {}
        x = x_dict['output']

        res = self.cls_loss(x, cls_target)
        loss_res.update(res)

        res = self.ml_loss(x, mask_pos, mask_neg, mask_tri)
        loss_res.update(res)

        return loss_res
