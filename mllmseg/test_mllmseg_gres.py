import argparse
import json
import os
import random
import re
from copy import deepcopy
from functools import partial
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

    # Calculate position_ids for packed dataset
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
    image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    assert (ret["input_ids"][0] == image_end_token_id).sum() == 1, f"image tokens are truncated, this dataset is {ds_name}"
    # 处理seg mask to_tensor resize
    gt_masks = []
    for gt_mask in masks_list:
        gt_mask = torch.from_numpy(np.array(gt_mask))
        gt_mask[gt_mask > 0.001] = 1.0
        gt_masks.append(gt_mask.float())
    gt_masks = torch.stack(gt_masks, dim=0)

    # Create the final return dictionary
    input_ids = ret["input_ids"]
    labels = ret["labels"]
    attention_mask = ret["attention_mask"]
    pixel_values = torch.concat(pixel_values_list, dim=0)
    image_flags = torch.tensor([1] * 1, dtype=torch.long)
    # bboxes_list = torch.stack(bboxes_list, dim=0)
    return input_ids, labels, attention_mask, pixel_values, image_flags, gt_masks, image_path_list, conversations


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
def evaluate_gres(args, dataset_name, base_image_dir):
    random.seed(args.seed)
    dataset = ValDataset(base_image_dir=base_image_dir, tokenizer=tokenizer, val_dataset=dataset_name, image_size=448)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, template_name="internvl2_5"),
    )

    inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    g_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    nt_tp_meter = AverageMeter("NT_TP", ":6.3f", Summary.SUM)
    nt_tn_meter = AverageMeter("NT_TN", ":6.3f", Summary.SUM)
    nt_fp_meter = AverageMeter("NT_FP", ":6.3f", Summary.SUM)
    nt_fn_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)

    for (
        input_ids,
        labels,
        attention_mask,
        pixel_values,
        image_flags,
        gt_masks,
        image_path_list,
        conversations,
    ) in tqdm(dataloader):
        torch.cuda.empty_cache()
        input_ids = input_ids.to(torch.int).cuda()
        pixel_values = pixel_values.to(torch.float16).cuda()
        gt_masks = gt_masks.to(torch.float16).cuda()
        input_ids_copy = input_ids[0].clone()

        output_dict = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            gt_masks=gt_masks,
            reeval=True,
            inference=True,
        )
        pred_masks = output_dict["pred_masks"]
        output_ids = output_dict["output_ids"][0]
        seg_or_rej_index = ((input_ids_copy == model.seg_token_idx) | (input_ids_copy == model.rej_token_idx)).nonzero(as_tuple=True)[0]
        pred_nts = output_ids[seg_or_rej_index] == model.rej_token_idx
        gt_masks = gt_masks[0]
        pred_masks = pred_masks[0]
        assert len(seg_or_rej_index) == len(gt_masks)
        assert len(pred_masks) == len(gt_masks)
        for b_idx, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=gt.shape[-2:], mode="bilinear").squeeze(0).squeeze(0)
            if gt.sum() < 1.0:  # empty target
                inter_i, union_i, _ = intersectionAndUnionGPU(pred.contiguous().clone(), gt.contiguous().clone(), K=2, ignore_index=255)
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                if pred_nts[b_idx]:
                    nt_tp_meter.update(1.0)
                    g_iou_meter.update(1.0)
                else:
                    nt_fn_meter.update(1.0)
                    g_iou_meter.update(0.0)
                    union_meter.update(union_i)
            else:
                if pred_nts[b_idx]:
                    nt_fp_meter.update(1.0)
                else:
                    nt_tn_meter.update(1.0)
                inter_i, union_i, _ = intersectionAndUnionGPU(pred.contiguous().clone(), gt.contiguous().clone(), K=2, ignore_index=255)
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                this_giou = inter_i / (union_i + 1e-8)
                inter_meter.update(inter_i)
                union_meter.update(union_i)
                g_iou_meter.update(this_giou)

    inter_meter.all_reduce()
    union_meter.all_reduce()
    g_iou_meter.all_reduce()
    nt_tp_meter.all_reduce()
    nt_tn_meter.all_reduce()
    nt_fp_meter.all_reduce()
    nt_fn_meter.all_reduce()

    N_acc = nt_tp_meter.sum / (nt_tp_meter.sum + nt_fn_meter.sum)  # for gt is empty, pred is empty
    T_acc = nt_tn_meter.sum / (nt_tn_meter.sum + nt_fp_meter.sum)  # for gt is target, pred is target
    g_iou = g_iou_meter.avg[1]
    c_iou = (inter_meter.sum / (union_meter.sum + 1e-10))[1]
    if torch.distributed.get_rank() == 0:
        log_stats = {"test_model:": args.checkpoint}
        log_stats["dataset_name"] = dataset_name
        log_stats["N_acc"] = round(N_acc * 100, 2)
        log_stats["T_acc"] = round(T_acc * 100, 2)
        log_stats["g_iou"] = round(g_iou * 100, 2)
        log_stats["c_iou"] = round(c_iou * 100, 2)
        print(log_stats)
        with open(os.path.join(args.checkpoint, "eval_log.txt"), mode="a") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="work_dirs/01_pretrain_lora16/checkpoint-44000_merged")
    parser.add_argument("--base_image_dir", type=str, default="/share/wangjingchao/gres_datasets")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--val_dataset", default="refcoco+|unc|testA", type=str)
    args = parser.parse_args()

    assert args.batch_size == 1, "Only batch size 1 is supported"
    val_datasets = [
        "grefcoco|unc|val",
        "grefcoco|unc|testA",
        "grefcoco|unc|testB",
    ]

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

    PATTERN = re.compile(r"\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*")
    model, tokenizer = load_model_and_tokenizer(args)

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    dec_params = sum(p.numel() for p in model.res_decoder.parameters()) / 1e6
    for val_dataset in val_datasets:
        evaluate_gres(args, val_dataset, args.base_image_dir)
