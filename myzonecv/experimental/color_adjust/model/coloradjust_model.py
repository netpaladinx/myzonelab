from myzonecv.core.model import BaseModel
from myzonecv.core.registry import BACKBONES, HEADS, LOSSES, POSTPROCESSORS
from myzonecv.core.utils import auto_fp16
from ..registry import COLORADJUST_MODELS


@COLORADJUST_MODELS.register_class('color_adjustor')
class ColorAdjustor(BaseModel):
    def __init__(self,
                 backbone,
                 head,
                 loss=None,
                 predict=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.backbone_cfg = backbone
        self.head_cfg = head
        self.loss_cfg = loss
        self.predict_cfg = predict

        self.backbone = BACKBONES.create(self.backbone_cfg)
        self.head = HEADS.create(self.head_cfg)
        self.loss = LOSSES.create(self.loss_cfg)
        self.predict = POSTPROCESSORS.create(self.predict_cfg)

        self.init_weights()

    def compute_loss(self, stats):
        """ stats (dict):
                'r_mean': bs
                'r_std': bs
                'g_mean': bs
                'g_std': bs
                'b_mean': bs
                'b_std': bs
                'brightness': bs
                'contrast': bs
                's_mean': bs
                's_std': bs
        """
        r_mean = stats['r_mean']
        r_std = stats['r_std']
        g_mean = stats['g_mean']
        g_std = stats['g_std']
        b_mean = stats['b_mean']
        b_std = stats['b_std']
        brt = stats['brightness']
        cont = stats['contrast']
        s_mean = stats['s_mean']
        s_std = stats['s_std']

        loss_res = self.loss(r_mean, r_std, g_mean, g_std, b_mean, b_std, brt, cont, s_mean, s_std)
        return loss_res

    @auto_fp16(apply_to=('inputs',))
    def forward_train(self, inputs, targets, batch_dict, **kwargs):
        """ inputs (dict)
                'input_stats': B x D (D=10)
            batch_dict:
                'img': list(C x Hi x Wi)
        """
        input_stats = inputs['input_stats']
        imgs = batch_dict['img']

        out = self.backbone(input_stats)
        out_imgs, out_stats, op_params = self.head(imgs, out)

        results = {}
        summary_keys = []

        loss_res = self.compute_loss(out_stats)
        summary_keys += self.collect_summary(loss_res)
        results.update(loss_res)

        results = self.merge_losses(results)
        results['summary_keys'] = summary_keys
        return results

    @auto_fp16(apply_to=('inputs',))
    def forward_predict(self, inputs, batch_dict=None, **kwargs):
        input_stats = inputs['input_stats']
        img = batch_dict['img']

        out = self.backbone(input_stats)
        out_img, out_stats, op_params = self.head(img, out)

        in_imgs_np, out_imgs_np, in_stats_np, out_stats_np, op_params_np = self.predict(batch_dict, self)

        results = {
            'in_img': img,       # list(torch.Tensor), CHW, float, 0~1
            'out_img': out_img,  # list(torch.Tensor), CHW, float, 0~1
            'in_stats': input_stats,
            'out_stats': out_stats,
            'op_params': op_params,
            'in_img_np': in_imgs_np,    # list(np.ndarry), HWC, ubyte, 0~255
            'out_img_np': out_imgs_np,  # list(np.ndarry), HWC, ubyte, 0~255
            'in_stats_np': in_stats_np,
            'out_stats_np': out_stats_np,
            'op_params_np': op_params_np,
            'intermeidate_keys': ('in_img', 'out_img', 'in_img_np', 'out_img_np')
        }
        return results
