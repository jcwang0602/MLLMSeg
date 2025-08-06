from copy import deepcopy
import glob
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from pycocotools.coco import COCO
from mllmseg_internvl.constants import ANSWER_LIST, SHORT_QUESTION_LIST_MODE4, ANSWER_LIST_MODE4_START, ANSWER_LIST_MODE4_TEMPLATE, ANSWER_LIST_MODE4_END, IMG_END_TOKEN
from mllmseg_internvl.dataset import build_transform
from mllmseg_internvl.dataset import preprocess, preprocess_internlm, preprocess_internvl2_5, preprocess_mpt, preprocess_phi3


def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(glob.glob(os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")))
    mapillary_images = [x.replace(".png", ".jpg").replace("v2.0/labels", "images") for x in mapillary_labels]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    with open("literals/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(os.listdir(os.path.join(base_image_dir, "ade20k/images", "training")))
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ade20k",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [x.replace(".jpg", ".png").replace("images", "annotations") for x in ade20k_images]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("literals/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(os.path.join(base_image_dir, "cocostuff", "train2017", "*.png"))
    cocostuff_images = [x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(os.path.join(base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"))
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json"))
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part


class SemSegDataset(TorchDataset):
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
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
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

        self.short_question_list = SHORT_QUESTION_LIST_MODE4
        self.answer_list = ANSWER_LIST

        self.ds_name = "sem_seg"

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {c: i for i, c in enumerate(self.data2classes["cocostuff"])}

        if self.no_sampling:
            len(self.sem_seg_datas) == 1

    def __len__(self):
        if self.no_sampling:
            ds = self.sem_seg_datas[0]
            return len(self.data2list[ds][0])
        else:
            return self.samples_per_epoch

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        return Image.open(image_path).convert("RGB")

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
        if self.no_sampling:
            ds = self.sem_seg_datas[0]
        else:
            ds = random.randint(0, len(self.sem_seg_datas) - 1)
            ds = self.sem_seg_datas[ds]

        if ds in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            # Get a positive example
            if not self.no_sampling:
                idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join("VOCdevkit", "VOC2010", "JPEGImages", file_name)
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)

            image = self.load_image(image_path)
            images = [image]

            transform = self.get_transform()
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)

            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self.__getitem__(0)
            if len(anns) >= self.num_classes_per_sample:
                sampled_anns = np.random.choice(anns, size=self.num_classes_per_sample, replace=False).tolist()
            else:
                sampled_anns = anns
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append((name, 0))
            n_pos_samples = len(sampled_anns)

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image, labels = self.data2list[ds]
            # Get a positive example
            if not self.no_sampling:
                idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255

            image = self.load_image(image_path)
            images = [image]

            transform = self.get_transform()
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)

            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            if len(classes) >= self.num_classes_per_sample:
                sampled_classes = np.random.choice(classes, size=self.num_classes_per_sample, replace=False).tolist()
            else:
                sampled_classes = classes
            n_pos_samples = len(sampled_classes)
            sampled_classes = [(sampled_class, 0) for sampled_class in sampled_classes]

        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."

        questions = []
        answers = []
        class_ids = []
        choice = np.random.randint(0, len(ANSWER_LIST_MODE4_START))
        question_template = random.choice(self.short_question_list)
        texts, is_rejs = [], []
        assert len(sampled_classes) > 0
        for sampled_cls in sampled_classes:
            text, is_rej = sampled_cls
            assert text != ""
            assert len(text.split("||")) == 1
            texts.append(text)
            is_rejs.append(is_rej)
            if ds in ["paco_lvis", "pascal_part"]:
                continue
            class_id = self.data2classes[ds].tolist().index(text)
            class_ids.append(class_id if not is_rej else -3)
        questions.append(question_template.format(class_name=", ".join(texts).lower()))
        ans_start, ans_template, ans_end = ANSWER_LIST_MODE4_START[choice], ANSWER_LIST_MODE4_TEMPLATE[choice], ANSWER_LIST_MODE4_END[choice]
        seg_token_parts = []
        for text, is_rej in zip(texts, is_rejs):
            added = ans_template.format(class_name=text)
            if is_rej:
                added = added.replace("SEG", "REJ")
            seg_token_parts.append(added)
        answers.append(ans_start + " " + ",".join(seg_token_parts) + ans_end)

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

        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for ann in sampled_anns:
                try:
                    masks.append(coco_api.annToMask(ann))
                except Exception as e:
                    print(e)
                    return self.__getitem__(0)
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        else:
            label = torch.from_numpy(label).long()
            masks = []
            for class_id in class_ids:
                masks.append(label == class_id)
            masks = torch.stack(masks, dim=0)

        assert masks.shape[0] == n_pos_samples

        # Calculate position_ids for packed dataset
        position_ids = ret["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret["input_ids"][0] == image_end_token_id).sum() == 1, f"image tokens are truncated, this dataset is {self.ds_name}"
        # for debug

        # 处理 seg mask to_tensor resize 这里默认mask数量和bbox数量一致
        gt_masks = []
        gt_bboxes = []

        for gt_mask in masks:
            gt_mask = torch.from_numpy(np.array(gt_mask))
            gt_mask[gt_mask > 0.001] = 1.0
            gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0).float(), size=(self.image_size, self.image_size), mode="nearest").squeeze(0).squeeze(0)
            gt_masks.append(gt_mask)
            gt_bboxes.append(torch.Tensor([0, 0, 0, 0]))

        gt_masks = torch.stack(gt_masks, axis=0)
        gt_bboxes = torch.stack(gt_bboxes, axis=0)
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
