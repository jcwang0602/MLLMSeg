from copy import deepcopy
import json
import os
import random
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from mllmseg_internvl.dataset import build_transform
from mllmseg_internvl.constants import DEFAULT_IMAGE_TOKEN, IMG_END_TOKEN
from mllmseg_internvl.dataset import preprocess, preprocess_internlm, preprocess_internvl2_5, preprocess_mpt, preprocess_phi3


def preprocess_multimodal(source):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
    return source


class VQADataset(TorchDataset):
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
        vqa_data="llava_instruct_150k",
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
        self.num_classes_per_sample = num_classes_per_sample  # this is not used in VQA tasks

        self.group_by_length = group_by_length
        self.use_packed_ds = use_packed_ds
        self.is_train = is_train
        self.pad2square = pad2square
        self.normalize_type = normalize_type

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.ds_name = "vqa"

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        print("vqa_data: ", len(self.vqa_data))

    def __len__(self):
        if self.no_sampling:
            return len(self.vqa_data)
        else:
            return self.samples_per_epoch

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

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        return Image.open(image_path).convert("RGB")

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square, normalize_type=self.normalize_type)
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
        if not self.no_sampling:
            idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        image = self.load_image(image_path)

        images = [image]

        transform = self.get_transform()
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        source = item["conversations"]
        conversations = preprocess_multimodal(source)

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

        # 因为视觉问答不需要进行分割，所以这次生成空的mask
        gt_masks = torch.rand(0, self.image_size, self.image_size)
        gt_bboxes = torch.rand(0, 4)
        do_seg = False

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
