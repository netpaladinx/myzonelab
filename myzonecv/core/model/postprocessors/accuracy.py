import torch
import torch.nn.functional as F

from ...registry import POSTPROCESSORS
from .base_process import BaseProcess


@POSTPROCESSORS.register_class('cls_accuracy')
class ClsAccuracy(BaseProcess):
    def __init__(self, topk=1, prob_thr=None, ignore_index=None, default='compute_accuracy', accuracy_name='acc'):
        """ topk: int (return one accuracy) or tuple(int) (returns multiple accuracies)
            prob_thr: probablity threshold to consider a prediction to be correct 
        """
        super().__init__(default)
        self.topk = topk
        self.prob_thr = prob_thr
        self.ignore_index = ignore_index
        self.accuracy_name = accuracy_name

    def compute_accuracy(self, pred, target):
        """ pred (logits): N x C x ...
            target: N x ...
        """
        assert pred.ndim == target.ndim + 1
        assert pred.shape[0] == target.shape[0]

        if isinstance(self.topk, int):
            topk = (self.topk,)
            return_single = True
        else:
            return_single = False

        n_pred = pred.shape[0]
        if n_pred == 0:
            accs = [pred.new_tensor(0.) for i in range(len(topk))]
            return {self.accuracy_name: accs[0]} if return_single else {f'{self.accuracy_name}s': accs}

        n_cls = pred.shape[0]
        maxk = min(max(topk), n_cls)

        with torch.no_grad():
            pred_prob = F.softmax(pred, dim=1)
            pred_prob, pred_label = pred_prob.topk(maxk, dim=1)  # N x maxk x ...
            pred_prob = pred_prob.transpose(0, 1)  # maxk x N x ...
            pred_label = pred_label.transpose(0, 1)  # maxk x N x ...

            correct = pred_label.eq(target)  # maxk x N x ...
            if self.prob_thr is not None:
                correct &= pred_prob >= self.prob_thr

            if self.ignore_index is not None:
                correct = correct[:, target != self.ignore_index]  # maxk x n_valid

            accs = []
            eps = torch.finfo(torch.float32).eps
            for k in topk:
                correct_num = correct[:k].sum()
                total_num = target[target != self.ignore_index].numel()
                accs.append(correct_num / (total_num + eps))
            return {self.accuracy_name: accs[0]} if return_single else {f'{self.accuracy_name}s': accs}


@POSTPROCESSORS.register_class('binary_accuracy')
class BinaryAccuracy(BaseProcess):
    def __init__(self, prob_thr=0.5, default='compute_accuracy', accuracy_name='acc'):
        """ thr: probablity threshold to consider a prediction to be correct 
        """
        super().__init__(default)
        self.prob_thr = prob_thr
        self.accuracy_name = accuracy_name

    def compute_accuracy(self, pred, target):
        """ pred (logits): N x ...
            target: N x ...
        """
        assert pred.shape == target.shape

        n_pred = pred.shape[0]
        if n_pred == 0:
            acc = pred.new_tensor(0.)
            return {self.accuracy_name: acc}

        with torch.no_grad():
            pred_prob = pred.sigmoid()
            pred_label = pred_prob > self.prob_thr
            target = target > self.prob_thr
            correct = pred_label.eq(target)  # N x ...

            eps = torch.finfo(torch.float32).eps
            correct_num = correct.sum()
            total_num = target.numel()
            pred_num = pred_label.sum()
            target_num = target.sum()
            tp_num = (correct & pred_label).sum()

            return {
                self.accuracy_name: correct_num / (total_num + eps),
                self.accuracy_name.replace('acc', 'prec'): tp_num / (pred_num + eps),
                self.accuracy_name.replace('acc', 'rec'): tp_num / (target_num + eps)
            }
