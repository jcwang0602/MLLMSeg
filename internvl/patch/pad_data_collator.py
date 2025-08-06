# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch

IGNORE_INDEX = -100


def pad_data_collator(features, pad_id=0):
    first = features[0]
    batch = {}

    batch_lens = [feat["input_ids"].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
        feat["input_ids"] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[: feat["labels"].shape[0]] = feat["labels"]
        feat["labels"] = temp_labels
        feat["attention_mask"] = feat["input_ids"].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


def concat_pad_data_collator(features, max_item_length=None, pad_id=0):
    first = features[0]
    batch = {}

    batch_lens = [feat["input_ids"].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
        feat["input_ids"] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[: feat["labels"].shape[0]] = feat["labels"]
        feat["labels"] = temp_labels
        feat["attention_mask"] = feat["input_ids"].ne(pad_id)

        if "position_ids" in feat:
            temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_position_ids[: feat["position_ids"].shape[0]] = feat["position_ids"]
            feat["position_ids"] = temp_position_ids

        if "loss_weight" in feat:
            temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
            temp_loss_weight[: feat["loss_weight"].shape[0]] = feat["loss_weight"]
            feat["loss_weight"] = temp_loss_weight

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids", "pixel_values", "image_flags") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ("pixel_values", "image_flags"):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch


def concat_pad_vgdata_collator(features, max_item_length=None, pad_id=0):
    first = features[0]
    batch = {}
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():  # 遍历第一个样本的所有键值对
        # 需要stack创建维度进行concat
        if k in ("input_ids", "labels", "attention_mask", "position_ids") and v is not None and not isinstance(v, str):  # 处理需要padding的张量数据
            temp_list = []  # 创建临时列表存储数据
            # 将多个样本的列表合并成一个列表
            for f in features:  # 遍历所有样本
                temp_list.extend(f[k])  # 将当前样本的数据添加到临时列表中
            # padding 到相同的长度
            batch[k] = torch.nn.utils.rnn.pad_sequence(temp_list, batch_first=True, padding_value=pad_id)  # 对列表中的序列进行padding
            if k == "input_ids":  # 如果是input_ids
                batch["attention_mask"] = batch["input_ids"].ne(pad_id)  # 创建attention mask
            # print(f"batch[{k}]:{batch[k].shape}")  # 打印当前batch的形状
        elif k in ("pixel_values", "gt_bboxes", "image_flags"):  # 处理图像相关的数据
            if isinstance(v, torch.Tensor):  # 如果是tensor类型
                batch[k] = torch.concat([f[k] for f in features])  # 直接连接
            elif isinstance(v, np.ndarray):  # 如果是numpy数组
                batch[k] = torch.concat(np.stack([f[k] for f in features]))  # 堆叠后连接
            else:  # 其他类型
                batch[k] = torch.concat([f[k] for f in features])  # 直接连接
            # print(f"batch[{k}]:{batch[k].shape}")  # 打印当前batch的形状
        elif k in ("image_sam",):  # 处理图像相关的数据
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
            # print(f"batch[{k}]:{batch[k].shape}")  # 打印当前batch的形状
        elif k in ("masks_sam", "gt_masks", "label_sam", "resize_sam", "do_seg"):  # 处理SAM模型相关的mask和label
            # 用于SAM，这里的形状不一样，只能放在列表中
            batch[k] = [f[k].float() if k == "masks_sam" else f[k] for f in features]  # 创建列表存储数据
        elif k in ("conversations",):  # 处理对话数据
            cnt = 0  # 初始化计数器
            offset_list = [0]  # 创建偏移量列表
            for f in features:  # 遍历所有样本
                cnt += len(f["conversations"])  # 累加对话长度
                offset_list.append(cnt)  # 添加新的偏移量
            batch["offset"] = torch.tensor(offset_list)  # 将偏移量转换为tensor
            batch["conversations"] = [f["conversations"] for f in features]  # 存储所有对话
            # print(f"offset:{batch['offset']}")  # 打印偏移量
            # print(batch["conversations"])
    return batch


def concat_pad_vgdata2_collator(features, max_item_length=None, pad_id=0):
    first = features[0]

    batch = {}
    batch_lens = []
    input_ids_list = []
    max_item_length = max_item_length or max(batch_lens)[0]

    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
        feat["input_ids"] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[: feat["labels"].shape[0]] = feat["labels"]
        feat["labels"] = temp_labels
        feat["attention_mask"] = feat["input_ids"].ne(pad_id)

        if "position_ids" in feat:
            temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_position_ids[: feat["position_ids"].shape[0]] = feat["position_ids"]
            feat["position_ids"] = temp_position_ids

        if "loss_weight" in feat:
            temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
            temp_loss_weight[: feat["loss_weight"].shape[0]] = feat["loss_weight"]
            feat["loss_weight"] = temp_loss_weight

        if "do_seg" in feat:
            feat["do_seg"] = torch.tensor(feat["do_seg"])

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        # 需要stack创建维度进行concat
        if (
            k
            not in (
                "input_ids",
                "labels",
                "label_ids",
                "pixel_values",
                "gt_masks",
                "gt_bboxes",
                "image_flags",
                "masks_sam",
                "label_sam",
                "conversations",
                "attention_mask",
            )
            and v is not None
            and not isinstance(v, str)
        ):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in (
            "input_ids",
            "labels",
            "attention_mask",
            "position_ids",
            "pixel_values",
            "image_sam",
            "resize_sam",
            "gt_masks",
            "gt_bboxes",
            "image_flags",
            "seg_label",
        ):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])

        elif k in ("masks_sam", "label_sam"):
            # 用于SAM，这里的形状不一样，只能放在列表中
            batch[k] = [f[k].float() if k == "masks_sam" else f[k] for f in features]
        elif k in ("input_ids_list", "labels", "attention_mask", "position_ids"):
            # 这几个长短不一，需要特殊处理
            batch[k] = [f[k].float() if k == "masks_sam" else f[k] for f in features]
        elif k in ("conversations",):
            cnt = 0
            offset_list = [0]
            for f in features:
                cnt += len(f["conversations"])
                offset_list.append(cnt)
            batch["offset"] = torch.tensor(offset_list)
            batch["conversations"] = [f["conversations"] for f in features]
    return batch


def dpo_concat_pad_data_collator(features, pad_id=0):
    first = features[0]
    batch = {}

    for prefix in ["chosen_", "rejected_"]:
        batch_lens = [feat[f"{prefix}input_ids"].shape[0] for feat in features]
        max_item_length = max(batch_lens)
        for idx in range(len(features)):
            feat = features[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[: feat[f"{prefix}input_ids"].shape[0]] = feat[f"{prefix}input_ids"]
            feat[f"{prefix}input_ids"] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[: feat[f"{prefix}labels"].shape[0]] = feat[f"{prefix}labels"]
            feat[f"{prefix}labels"] = temp_labels
            feat[f"{prefix}attention_mask"] = feat[f"{prefix}input_ids"].ne(pad_id)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("pixel_values", "image_flags") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ("pixel_values", "image_flags"):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch
