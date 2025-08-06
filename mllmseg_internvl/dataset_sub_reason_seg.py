from copy import deepcopy
import glob
import json
import os
import random

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from mllmseg_internvl.constants import (
    ANSWER_LIST,
    DEFAULT_IMAGE_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    EXPLANATORY_QUESTION_LIST,
    LONG_QUESTION_LIST,
    SHORT_QUESTION_LIST,
)
from mllmseg_internvl.dataset import build_transform
from mllmseg_internvl.dataset import (
    preprocess,
    preprocess_internlm,
    preprocess_internvl2_5,
    preprocess_mpt,
    preprocess_phi3,
)


def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.size

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence


class ReasonSegDataset(TorchDataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        is_train,
        pad2square,
        normalize_type,
        template_name,
        num_image_token,
        base_image_dir,
        tokenizer,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        no_sampling=False,
        dynamic_image_size=False,
        group_by_length=False,
        use_packed_ds=False,
    ):
        self.num_image_token = num_image_token
        self.dynamic_image_size = dynamic_image_size
        self.template_name = template_name
        self.no_sampling = no_sampling
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.group_by_length = group_by_length
        self.use_packed_ds = use_packed_ds
        self.is_train = is_train
        self.pad2square = pad2square
        self.normalize_type = normalize_type

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.explanatory = explanatory

        self.ds_name = "reason_seg"

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            images_split = glob.glob(os.path.join(base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"))
            images.extend(images_split)
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))

        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    base_image_dir,
                    "reason_seg",
                    reason_seg_data,
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))

    def __len__(self):
        if self.no_sampling:
            return len(self.reason_seg_data[0])
        else:
            return self.samples_per_epoch

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        return Image.open(image_path).convert("RGB")

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == "Hermes-2":
            preprocess_function = preprocess_mpt
        elif self.template_name == "internlm2-chat":
            preprocess_function = preprocess_internlm
        elif self.template_name == "phi3-chat":
            preprocess_function = preprocess_phi3
        elif self.template_name == "internvl2_5":
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def preprocess_sam(self, x: torch.Tensor) -> torch.Tensor:
        # For SAM
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        if not self.no_sampling:
            idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        image = self.load_image(image_path)
        images = [image]
        # preprocess image for clip
        transform = self.get_transform()
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        num_patches = pixel_values.size(0)

        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(list(range(len(sents))), size=self.num_classes_per_sample, replace=False)
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [(mask == 1).astype(np.float32) for _ in range(len(sampled_inds))]

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2
            else:
                choice = random.randint(0, 1)

        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] token + text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text + " {}".format(random.choice(self.explanatory_question_list))
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

            assert len(questions) == len(answers), "问题和答案数量不匹配，麻烦大了"

            conversations = []
            # TODO: 在这里，需要将多个问答直接切割成一问一答的形式
            for i in range(len(questions)):
                conversations.append(
                    [
                        {"from": "human", "value": questions[i]},
                        {"from": "gpt", "value": answers[i]},
                    ]
                )

        preprocess_function = self.get_preprocess_function()
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        position_ids_list = []

        for conversation in conversations:
            ret = preprocess_function(
                self.template_name,
                [deepcopy(conversation)],
                self.tokenizer,
                [self.num_image_token * num_patches],
                group_by_length=self.group_by_length,
                use_packed_ds=self.use_packed_ds,
                ds_name=self.ds_name,
            )

            # Calculate position_ids for packed dataset
            position_ids = ret["attention_mask"].long().cumsum(-1) - 1
            position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
            image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
            assert (ret["input_ids"][0] == image_end_token_id).sum() == 1, f"image tokens are truncated, this dataset is {self.ds_name}"

            input_ids_list.append(ret["input_ids"][0])
            labels_list.append(ret["labels"][0])
            attention_mask_list.append(ret["attention_mask"][0])
            position_ids_list.append(position_ids[0])

        image_name = image_path.split("/")[-1]

        # # 处理seg mask to_tensor resize
        if self.explanatory != -1 and image_name in self.img_to_explanation and choice == 2:
            gt_masks = torch.rand(0, self.image_size, self.image_size)
            # gt_masks_new = torch.rand(0, self.image_size, self.image_size)
            gt_bboxes = torch.rand(0, 4)
            do_seg = False
        else:
            # gt_masks_new = F.interpolate(masks_sam.unsqueeze(1).float(), size=(self.image_size, self.image_size), mode="nearest").squeeze(1)
            # gt_masks_new = gt_masks_new.bool().byte()

            gt_masks = []
            gt_bboxes = []
            for gt_mask in sampled_masks:
                gt_mask = torch.from_numpy(np.array(gt_mask))
                gt_mask[gt_mask > 0.001] = 1.0
                gt_mask = (
                    F.interpolate(
                        gt_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(self.image_size, self.image_size),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                gt_masks.append(gt_mask)
                gt_bboxes.append(torch.Tensor([0, 0, 0, 0]))

            gt_masks = torch.stack(gt_masks, axis=0)
            gt_bboxes = torch.stack(gt_bboxes, axis=0)
            do_seg = True

        ret = dict(
            input_ids=input_ids_list,
            labels=labels_list,
            attention_mask=attention_mask_list,
            position_ids=position_ids_list,
            pixel_values=pixel_values,
            conversations=conversations,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            gt_bboxes=gt_bboxes,
            gt_masks=gt_masks,
            do_seg=do_seg,
        )
        return ret
