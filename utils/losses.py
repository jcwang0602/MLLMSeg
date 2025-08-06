# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Written by Zhuofan Xia
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area


def iou_loss(pred_iou: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor, num_masks: float):
    pred_iou = pred_iou.to(torch.float32).sigmoid()
    pred_mask_ = pred_mask.detach().clone()
    target_mask_ = target_mask.detach().clone()
    inter = (pred_mask_ * target_mask_).sum()
    union = pred_mask_.sum() + target_mask_.sum() - inter
    gt_iou = inter / (union + 1e-8)

    iou_loss = ((gt_iou - pred_iou) ** 2).sum() / (num_masks + 1e-8)
    return iou_loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def mask_iou(inputs, targets):
    """
    计算像素级别的IOU均值
    Args:
        inputs: 预测的mask, shape为[B, H, W]
        targets: 真实的mask, shape为[B, H, W]
    Returns:
        iou: 平均IOU值
    """
    # 展平空间维度
    inputs = inputs.flatten(1, 2)  # [B, H*W]
    targets = targets.flatten(1, 2)  # [B, H*W]

    # 计算交集
    inter = (inputs * targets).sum(-1)  # [B]

    # 计算并集
    union = inputs.sum(-1) + targets.sum(-1) - inter  # [B]

    # 计算每个样本的IOU
    iou = (inter + 1e-8) / (union + 1e-8)  # [B]

    # 返回batch内的平均IOU
    batch_size = inputs.shape[0]
    return iou.sum() / batch_size


def dice_loss_plvl(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    batch_size = inputs.shape[0]
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    # Ensure inputs and targets have the same dtype
    if inputs.dtype != targets.dtype:
        targets = targets.to(dtype=inputs.dtype)

    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / batch_size


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    batch_size = inputs.shape[0]
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    # Ensure inputs and targets have the same dtype
    if inputs.dtype != targets.dtype:
        targets = targets.to(dtype=inputs.dtype)

    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / batch_size


def focal_loss(prediction, target, alpha=2, beta=4):
    positive_index = target.eq(1).float()
    negative_index = target.lt(1).float()

    negative_weights = torch.pow(1 - target, beta)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    prediction = torch.clamp(prediction, 1e-12)

    positive_loss = torch.log(prediction) * torch.pow(1 - prediction, alpha) * positive_index
    # if torch.isnan(positive_loss).any() or torch.isinf(positive_loss).any():
    #     print("positive_loss is {}, stopping training, FocalLoss".format(positive_loss))
    epsilon = 1e-5
    negative_loss = torch.log(1 - prediction + epsilon) * torch.pow(prediction, alpha) * negative_weights * negative_index

    num_positive = positive_index.float().sum()
    positive_loss = positive_loss.sum()
    negative_loss = negative_loss.sum()

    if num_positive == 0:
        loss = -negative_loss
    elif torch.isnan(negative_loss).any() or torch.isinf(negative_loss).any():
        loss = -positive_loss / num_positive
        print(f"negative_loss is {negative_loss}")
    else:
        loss = -(positive_loss + negative_loss) / num_positive
    return loss


class CenterNetHeatMap(object):
    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = CenterNetHeatMap.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            CenterNetHeatMap.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        # box_tensor = torch.Tensor(box_size)
        box_tensor = box_size
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m : m + 1, -n : n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CenterNetHeatMap.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top : y + bottom, x - left : x + right] = masked_fmap
        # return fmap


def generate_heatmap(bboxes, patch_size=320, stride=16):
    """
    Generate ground truth heatmap same as CenterNet
    Args:
        bboxes (torch.Tensor): shape of [num_search, bs, 4]

    Returns:
        gaussian_maps: list of generated heatmap

    """
    gaussian_maps = []
    heatmap_size = patch_size // stride
    for single_patch_bboxes in bboxes:
        bs = single_patch_bboxes.shape[0]
        gt_scoremap = torch.zeros(bs, heatmap_size, heatmap_size)
        classes = torch.arange(bs).to(torch.long)
        bbox = single_patch_bboxes * heatmap_size
        wh = bbox[:, 2:]
        centers_int = (bbox[:, :2]).round()
        CenterNetHeatMap.generate_score_map(gt_scoremap, classes, wh, centers_int, 0.7)
        gaussian_maps.append(gt_scoremap.to(bbox.device))
    return gaussian_maps


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou_conv(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)  # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1]  # (N,)

    return iou - (area - union) / area, iou


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou_conv(boxes1, boxes2)
    return (1 - giou).mean(), iou
