from copy import deepcopy
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from pycocotools import mask
from PIL import Image
from mllmseg_internvl.refzom import REFZOM_REFER
from mllmseg_internvl.grefer import G_REFER
from mllmseg_internvl.refer import REFER
from mllmseg_internvl.constants import ANSWER_LIST_MODE4_START, ANSWER_LIST_MODE4_TEMPLATE, ANSWER_LIST_MODE4_END, IMG_END_TOKEN, SHORT_QUESTION_LIST_MODE4, ANSWER_LIST
from mllmseg_internvl.dataset import build_transform
from mllmseg_internvl.dataset import (
    preprocess,
    preprocess_internlm,
    preprocess_internvl2_5,
    preprocess_mpt,
    preprocess_phi3,
)


class ReferSegDataset(TorchDataset):
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
        num_classes_per_sample: int = 5,
        exclude_val=False,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        no_sampling=False,
        dynamic_image_size=False,
        group_by_length=False,
        use_packed_ds=False,
    ):
        self.num_image_token = num_image_token
        self.dynamic_image_size = dynamic_image_size
        self.template_name = template_name
        # 这里每个类型的数据集一般都是多个，所以这个没啥用
        # self.no_sampling = no_sampling
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

        self.short_question_list = SHORT_QUESTION_LIST_MODE4
        self.answer_list = ANSWER_LIST

        self.ds_name = "refer_seg"

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split("||")  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.no_sampling = no_sampling

        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            elif ds == "refzom":
                refer_api = REFZOM_REFER(DATA_DIR, ds)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(DATA_DIR, "images/saiapr_tc-12", item["file_name"])
                else:
                    item["file_name"] = os.path.join(DATA_DIR, "images/mscoco/images/train2014", item["file_name"])
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print("dataset {} (refs {}) (train split) has {} images and {} annotations.".format(ds, splitBy, len(refer_seg_ds["images"]), len(refer_seg_ds["annotations"])))

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [ref]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

        if self.no_sampling:
            assert len(self.refer_seg_ds_list) == 1

    def __len__(self):
        if self.no_sampling:
            ds = self.refer_seg_ds_list[0]
            return len(self.refer_seg_data[ds]["images"])
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

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        if not self.no_sampling:
            idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        n_pos_to_sample = self.num_classes_per_sample

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= n_pos_to_sample:
            sampled_inds = np.random.choice(list(range(len(sents))), size=n_pos_to_sample, replace=False)
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]

        masks = []
        bboxes = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                else:
                    m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                        else:
                            if isinstance(ann["segmentation"], dict):
                                rle = ann["segmentation"]
                                assert isinstance(rle["counts"], list)
                                # convert to compressed RLE
                                rle = mask.frPyObjects(rle, image_info["height"], image_info["width"])
                            else:
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            m = mask.decode(rle)
                            if m.ndim < 3:
                                assert m.ndim == 2
                                m = m[..., np.newaxis]
                            m = np.sum(m, axis=2).astype(np.uint8)  # convert to np.uint8
                            m = m
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            # 处理 bbox
            bbox = np.array(ann["bbox"], dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
            bboxes.append(bbox)

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                masks.append(m)
                continue

            if isinstance(ann["segmentation"][0], list):  # polygon
                rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        image = self.load_image(image_path)
        images = [image]

        transform = self.get_transform()
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."

        questions = []
        answers = []
        choice = np.random.randint(0, len(ANSWER_LIST_MODE4_START))
        question_template = random.choice(self.short_question_list)
        texts = []
        for text in sampled_sents:
            text = text.strip()
            assert text != ""
            assert len(text.split("||")) == 1
            texts.append(text.strip("."))
        questions.append(question_template.format(class_name=", ".join(texts).lower()))
        ans_start, ans_template, ans_end = (ANSWER_LIST_MODE4_START[choice], ANSWER_LIST_MODE4_TEMPLATE[choice], ANSWER_LIST_MODE4_END[choice])
        seg_token_parts = []
        all_rej = True

        # 在这里。把grefcoco中的REJ的mask去掉
        for t_idx, t in enumerate(texts):
            output_cls_prompt = ans_template.format(class_name=t)
            if masks[t_idx].sum() < 1.0:
                output_cls_prompt = output_cls_prompt.replace("SEG", "REJ")
            else:
                all_rej = False
            seg_token_parts.append(output_cls_prompt)
        if all_rej:
            return self.__getitem__(0)

        answers.append(ans_start + " " + ", ".join(seg_token_parts) + ans_end)

        assert len(questions) == 1 and len(answers) == 1, "问题和答案不只是1， 麻烦大了"

        conversations = []
        conversations = [
            {"from": "human", "value": questions[0]},
            {"from": "gpt", "value": answers[0]},
        ]

        preprocess_function = self.get_preprocess_function()
        ret = preprocess_function(
            self.template_name,
            [deepcopy(conversations)],
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

        # 处理seg mask to_tensor resize
        gt_masks = []
        for gt_mask in masks:
            gt_mask = torch.from_numpy(np.array(gt_mask))
            gt_mask[gt_mask > 0.001] = 1.0
            gt_mask = gt_mask.float()
            gt_masks.append(gt_mask)

        gt_masks = torch.stack(gt_masks, axis=0)
        gt_bboxes = torch.from_numpy(np.array(bboxes))
        do_seg = True

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"],
            labels=ret["labels"],
            attention_mask=ret["attention_mask"],
            position_ids=position_ids,
            pixel_values=pixel_values,
            conversations=[conversations],
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            gt_bboxes=gt_bboxes,
            gt_masks=gt_masks,
            do_seg=do_seg,
        )
        return ret
