import argparse
import json
import os
import random
from copy import deepcopy
from functools import partial
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from mllmseg.mllmseg_internvl import MLLMSeg
from mllmseg.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN
from mllmseg.dataset import preprocess, preprocess_internlm, preprocess_internvl2_5, preprocess_mpt, preprocess_phi3
from mllmseg.dataset_vg import ValDataset
from utils import AverageMeter, Summary


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = MLLMSeg.from_pretrained(
        args.checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        data_type=torch.bfloat16,
        tokenizer=tokenizer,
        init_decoder=True,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    rej_token_idx = tokenizer.convert_tokens_to_ids("[REJ]")
    seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    model.img_context_token_id = img_context_token_id
    model.seg_token_idx = seg_token_idx
    model.rej_token_idx = rej_token_idx
    return model, tokenizer


def intersectionAndUnionGPU(pred_masks, gt_masks, K, ignore_index=255):
    output = torch.ones(pred_masks.shape)
    output[pred_masks < 0.5] = 0
    target = torch.ones(gt_masks.shape)
    target[gt_masks <= 0.0] = 0
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape, f"output_shape = {output.shape}, target_shape = {target.shape}"
    output = output.reshape(-1)
    target = target.reshape(-1)
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def mask_to_bbox(mask):
    """
    将mask转换为外接矩形框 [x1, y1, x2, y2]
    """
    # 找到mask中非零像素的坐标
    coords = torch.nonzero(mask > 0.5)
    if coords.shape[0] == 0:
        return torch.tensor([0, 0, 0, 0], dtype=torch.float32)

    # 计算边界框 - 修复坐标顺序问题
    # coords的形状是 [N, 2]，其中第一列是y坐标，第二列是x坐标
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    return torch.tensor([x_min.item(), y_min.item(), x_max.item(), y_max.item()], dtype=torch.float32)


def calculate_bbox_iou(bbox1, bbox2):
    """
    计算两个边界框的IoU
    bbox格式: [x1, y1, x2, y2]
    """
    # 计算交集区域
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 计算并集区域
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_rec_bbox_iou(pred_masks, gt_masks, gt_bboxes, iou_threshold=0.5):
    success_count = 0
    total_count = 0
    for pred_mask, gt_mask, gt_bbox in zip(pred_masks, gt_masks, gt_bboxes):
        pred_mask_bbox = mask_to_bbox(pred_mask)
        gt_bbox[2] = gt_bbox[0] + gt_bbox[2]
        gt_bbox[3] = gt_bbox[1] + gt_bbox[3]
        iou = calculate_bbox_iou(pred_mask_bbox, gt_bbox)
        if iou > iou_threshold:
            success_count += 1
        total_count += 1
    return success_count, total_count


def collate_fn(batch, tokenizer=None, template_name="llava_v1", ds_name="test"):
    if template_name == "Hermes-2":
        preprocess_function = preprocess_mpt
    elif template_name == "internlm2-chat":
        preprocess_function = preprocess_internlm
    elif template_name == "phi3-chat":
        preprocess_function = preprocess_phi3
    elif template_name == "internvl2_5":
        preprocess_function = preprocess_internvl2_5
    else:
        preprocess_function = preprocess

    image_path_list = []
    pixel_values_list = []
    conversation_list = []
    masks_list = []
    bboxes_list = []
    cnt = 0
    for image_path, pixel_values, conversations, masks, bboxes in batch:
        image_path_list.append(image_path)
        pixel_values_list.append(pixel_values)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        bboxes_list.append(bboxes)
        cnt += len(conversations)

    ret = preprocess_function(
        template_name,
        [deepcopy(conversation_list)],
        tokenizer,
        [model.num_image_token],
        group_by_length=False,
        use_packed_ds=False,
        ds_name="test",
    )

    image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    assert (ret["input_ids"][0] == image_end_token_id).sum() == 1, f"image tokens are truncated, this dataset is {ds_name}"
    # 处理seg mask to_tensor resize
    gt_masks = []
    for gt_mask in masks_list:
        gt_mask = torch.from_numpy(np.array(gt_mask))
        gt_mask[gt_mask > 0.001] = 1.0
        gt_masks.append(gt_mask.float())
    gt_masks = torch.stack(gt_masks, dim=0)

    input_ids = ret["input_ids"]
    labels = ret["labels"]
    attention_mask = ret["attention_mask"]
    pixel_values = torch.concat(pixel_values_list, dim=0)
    image_flags = torch.tensor([1] * 1, dtype=torch.long)
    bboxes_list = torch.from_numpy(np.stack(bboxes_list, axis=0))
    return input_ids, labels, attention_mask, pixel_values, image_flags, gt_masks, image_path_list, bboxes_list, conversations


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


@torch.no_grad()
def evaluate(args, dataset_name, base_image_dir):
    random.seed(args.seed)
    dataset = ValDataset(base_image_dir=base_image_dir, tokenizer=tokenizer, val_dataset=dataset_name, image_size=448)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, template_name="internvl2_5"),
    )

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    rec_success_meter = AverageMeter("REC_Success", ":6.3f", Summary.SUM)
    rec_total_meter = AverageMeter("REC_Total", ":6.3f", Summary.SUM)

    model.eval()
    total_num = 0
    pred_masks_list = []
    gt_masks_list = []
    for (
        input_ids,
        labels,
        attention_mask,
        pixel_values,
        image_flags,
        gt_masks,
        image_path_list,
        bboxes_list,
        conversations,
    ) in tqdm(dataloader):
        torch.cuda.empty_cache()
        input_ids = input_ids.to(torch.int).cuda()
        pixel_values = pixel_values.to(torch.float16).cuda()
        gt_masks = gt_masks.to(torch.float16).cuda()
        bboxes_list = bboxes_list.to(torch.float16).cuda()
        output_dict = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            gt_masks=gt_masks,
            reeval=False,
            inference=True,
        )
        pred_masks = output_dict["pred_masks"]
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        batch_pred_masks = []
        batch_gt_masks = []
        batch_gt_bboxes = []
        for idx, (mask_i, output_i, bbox_i) in enumerate(zip(gt_masks[0], pred_masks[0], bboxes_list[0])):
            output_i = F.interpolate(output_i.unsqueeze(0).unsqueeze(0).float(), size=mask_i.shape[-2:], mode="bilinear").squeeze(0).squeeze(0)
            batch_pred_masks.append(output_i.contiguous().clone())
            batch_gt_masks.append(mask_i.contiguous().clone())
            batch_gt_bboxes.append(bbox_i.tolist())
            pred_masks_list.append(output_i.contiguous().clone())
            gt_masks_list.append(mask_i.contiguous().clone())

            intersection_i, union_i, _ = intersectionAndUnionGPU(output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255)
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0
        batch_pred_masks = torch.stack(batch_pred_masks, dim=0)
        batch_gt_masks = torch.stack(batch_gt_masks, dim=0)
        batch_gt_bboxes = torch.tensor(batch_gt_bboxes)
        rec_success, rec_total = evaluate_rec_bbox_iou(batch_pred_masks, batch_gt_masks, batch_gt_bboxes, iou_threshold=0.5)
        rec_success_meter.update(rec_success)
        rec_total_meter.update(rec_total)

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / gt_masks.shape[1]
        intersection_meter.update(intersection), union_meter.update(union), acc_iou_meter.update(acc_iou)

        total_num += gt_masks.shape[1]

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    rec_success_meter.all_reduce()
    rec_total_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    rec_accuracy = rec_success_meter.sum / (rec_total_meter.sum)
    if torch.distributed.get_rank() == 0:
        log_stats = {"test_model:": args.checkpoint}
        log_stats["dataset_name"] = dataset_name
        log_stats["ciou"] = round(ciou.item() * 100, 2)
        log_stats["giou"] = round(giou.item() * 100, 2)
        log_stats["rec_bbox_iou_0.5"] = round(rec_accuracy * 100, 2)
        print(log_stats)
        with open(os.path.join(args.checkpoint, "eval_log.txt"), mode="a") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="work_dirs/01_pretrain_lora16/checkpoint-44000_merged")
    parser.add_argument("--base_image_dir", type=str, default="/share/wangjingchao/gres_datasets")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    val_datasets = [
        "refcoco|unc|val",
        "refcoco|unc|testA",
        "refcoco|unc|testB",
        "refcoco+|unc|val",
        "refcoco+|unc|testA",
        "refcoco+|unc|testB",
        "refcocog|umd|val",
        "refcocog|umd|test",
    ]

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))
    torch.cuda.empty_cache()
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    dec_params = sum(p.numel() for p in model.res_decoder.parameters()) / 1e6
    for val_dataset in val_datasets:
        evaluate(args, val_dataset, args.base_image_dir)
