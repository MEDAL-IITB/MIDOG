from torch.autograd import Variable
from object_detection_fastai.helper.object_detection_helper import *
from torch import nn
import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, inp, target, feature_layers=[1], style_layers=[]):
        if inp.shape[1] != 3:
            inp = inp.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        inp = (inp-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            inp = self.transform(inp, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = inp
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)
        return loss

class my_RetinaNetFocalLossDAA(nn.Module):

    def __init__(self, anchors: Collection[float], gamma: float = 2., alpha: float = 0.25, pad_idx: int = 0,
                 reg_loss: LossFunction = F.smooth_l1_loss, domain_weight: float=0.001, n_domains = 2):
        super().__init__()
        self.gamma, self.alpha, self.pad_idx, self.reg_loss = gamma, alpha, pad_idx, reg_loss
        self.anchors = anchors
        self.metric_names = ['BBloss', 'focal_loss', 'domain_loss', 'total', 'acc', 'VGGloss']
        self.domain_weight = domain_weight
        self.n_domains = n_domains
        self.rec_loss=VGGPerceptualLoss().cuda()
        self.domain_loss=nn.CrossEntropyLoss()

    def _unpad(self, bbox_tgt, clas_tgt):
        i = torch.min(torch.nonzero(clas_tgt - self.pad_idx)) if sum(clas_tgt)>0 else 0
        return tlbr2cthw(bbox_tgt[i:]), clas_tgt[i:] - 1 + self.pad_idx

    def _focal_loss(self, clas_pred, clas_tgt):
        encoded_tgt = encode_class(clas_tgt, clas_pred.size(1))
        ps = torch.sigmoid(clas_pred)
        weights = Variable(encoded_tgt * (1 - ps) + (1 - encoded_tgt) * ps)
        alphas = (1 - encoded_tgt) * self.alpha + encoded_tgt * (1 - self.alpha)
        weights.pow_(self.gamma).mul_(alphas)
        clas_loss = F.binary_cross_entropy_with_logits(clas_pred, encoded_tgt, weights, reduction='sum')
        return clas_loss

    def _one_loss(self, clas_pred, bbox_pred, clas_tgt, bbox_tgt):
        bbox_tgt, clas_tgt = self._unpad(bbox_tgt, clas_tgt)
        matches = match_anchors(self.anchors, bbox_tgt)
        bbox_mask = matches >= 0
        if bbox_mask.sum() != 0:
            bbox_pred = bbox_pred[bbox_mask]
            bbox_tgt = bbox_tgt[matches[bbox_mask]]
            bb_loss = self.reg_loss(bbox_pred, bbox_to_activ(bbox_tgt, self.anchors[bbox_mask]))
        else:
            bb_loss = 0.
        matches.add_(1)
        clas_tgt = clas_tgt + 1
        clas_mask = matches >= 0
        clas_pred = clas_pred[clas_mask]
        clas_tgt = torch.cat([clas_tgt.new_zeros(1).long(), clas_tgt])
        clas_tgt = clas_tgt[matches[clas_mask]]
        return bb_loss, self._focal_loss(clas_pred, clas_tgt) / torch.clamp(bbox_mask.sum(), min=1.)

    # def domain_focal_loss(self, clas_pred, clas_tgt, alpha=0.25, gamma=2.):
    #     ce_loss = F.cross_entropy(clas_pred, clas_tgt)
    #     pt = torch.exp(-ce_loss)
    #     focal_loss = torch.mul(alpha,((1 - pt) ** gamma * ce_loss))
    #     return focal_loss



    def forward(self, output, bbox_tgts, clas_tgts, domain_tgts):
        clas_preds, bbox_preds, pre_output, sizes = output
        if bbox_tgts.device != self.anchors.device:
            self.anchors = self.anchors.to(clas_preds.device)
        bb_loss = torch.tensor(0, dtype=torch.float32).to(clas_preds.device)
        focal_loss = torch.tensor(0, dtype=torch.float32).to(clas_preds.device)
        
        pred_x, pred_label, org_img=pre_output

        r_loss = self.rec_loss(pred_x,org_img)
        # r_loss = torch.tensor(0).cuda()
        d_loss = self.domain_loss(pred_label,domain_tgts)

        # print(r_loss)
        # exit()
        
        acc = torch.true_divide(sum(torch.argmax(pred_label, dim=1) == domain_tgts), domain_tgts.size(0))

        for cp, bp, ct, bt, dt in zip(clas_preds, bbox_preds, clas_tgts,bbox_tgts, domain_tgts):
            if dt != 2:
                bb, focal = self._one_loss(cp, bp, ct, bt)
                bb_loss += bb
                focal_loss += focal
        
        total_loss = (bb_loss + focal_loss)/clas_tgts[domain_tgts!=2].size(0) + 10*r_loss - self.domain_weight * d_loss
        self.metrics = dict(zip(self.metric_names, [bb_loss / clas_tgts[domain_tgts!=2].size(0), 
                                                    focal_loss / clas_tgts[domain_tgts!=2].size(0),
                                                    d_loss,
                                                    total_loss,
                                                    acc,
                                                    r_loss]))
        return (bb_loss + focal_loss)/clas_tgts[domain_tgts!=2].size(0) + 10*r_loss + self.domain_weight * d_loss
