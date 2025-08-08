import logging
import math
import os
import warnings
from typing import Dict
import glob
import numpy as np
from pycocotools import mask
from mllmseg.constants import DEFAULT_IMAGE_TOKEN
from mllmseg.dataset_sub_reason_seg import ReasonSegDataset
from mllmseg.dataset_sub_refer_seg import ReferSegDataset
from mllmseg.dataset_sub_sem_seg import SemSegDataset
from mllmseg.dataset_sub_vqa import VQADataset
from mllmseg.dataset_sub_reason_seg import get_mask_from_json
import torch
from mllmseg.dataset import ConcatDataset, WeightedConcatDataset
from mllmseg.dataset_packed import PackedDataset
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset

from mllmseg.grefer import G_REFER
from mllmseg.refer import REFER
from mllmseg.refzom import REFZOM_REFER
from mllmseg.dataset import build_transform

# Set constants for image processing and logging
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        tokenizer,
        tcs_loader,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method="rand",  # for video data
        repeat_time=1,
        normalize_type="imagenet",
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
        samples_per_epoch=50000,
        dataset="",
        sem_seg_data="",
        refer_seg_data="",
        refer_det_data="",
        vqa_data="",
        reason_seg_data="",
        no_sampling=False,
        base_image_dir="",
    ):
        super(LazySupervisedDataset, self).__init__()
        self.template_name = template_name
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.no_sampling = no_sampling
        self.num_image_token = num_image_token

        base_image_dir = base_image_dir
        dataset = dataset
        self.datasets = dataset.split("||")
        if len(self.datasets) > 1:
            sample_rate = np.array([9, 6, 3, 1])
            self.sample_rate = sample_rate / sample_rate.sum()
        else:
            self.sample_rate = [1]
        print(f"self.sample_rate:{self.sample_rate}")
        print(f"self.no_sampling:{self.no_sampling}")
        self.samples_per_epoch = samples_per_epoch
        precision = "fp32"
        num_classes_per_sample = 5
        exclude_val = False
        explanatory = 0.1
        self.all_datasets = []
        self.length = [1024]

        for dataset in self.datasets:
            if dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        is_train=self.is_train,
                        pad2square=self.pad2square,
                        normalize_type=self.normalize_type,
                        template_name=template_name,
                        num_image_token=self.num_image_token,
                        base_image_dir=base_image_dir,
                        tokenizer=tokenizer,
                        samples_per_epoch=samples_per_epoch,
                        precision=precision,
                        image_size=image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        refer_seg_data=refer_seg_data,
                        no_sampling=self.no_sampling,
                        dynamic_image_size=dynamic_image_size,
                        group_by_length=group_by_length,
                        use_packed_ds=use_packed_ds,
                    )
                )
            elif dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        is_train=self.is_train,
                        pad2square=self.pad2square,
                        normalize_type=self.normalize_type,
                        template_name=template_name,
                        num_image_token=self.num_image_token,
                        base_image_dir=base_image_dir,
                        tokenizer=tokenizer,
                        samples_per_epoch=samples_per_epoch,
                        precision=precision,
                        image_size=image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        sem_seg_data=sem_seg_data,
                        no_sampling=self.no_sampling,
                        dynamic_image_size=dynamic_image_size,
                        group_by_length=group_by_length,
                        use_packed_ds=use_packed_ds,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        is_train=self.is_train,
                        pad2square=self.pad2square,
                        normalize_type=self.normalize_type,
                        template_name=template_name,
                        num_image_token=self.num_image_token,
                        base_image_dir=base_image_dir,
                        tokenizer=tokenizer,
                        samples_per_epoch=samples_per_epoch,
                        precision=precision,
                        image_size=image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        vqa_data=vqa_data,
                        no_sampling=self.no_sampling,
                        dynamic_image_size=dynamic_image_size,
                        group_by_length=group_by_length,
                        use_packed_ds=use_packed_ds,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        is_train=self.is_train,
                        pad2square=self.pad2square,
                        normalize_type=self.normalize_type,
                        template_name=template_name,
                        num_image_token=self.num_image_token,
                        base_image_dir=base_image_dir,
                        tokenizer=tokenizer,
                        samples_per_epoch=samples_per_epoch,
                        precision=precision,
                        image_size=image_size,
                        num_classes_per_sample=num_classes_per_sample,
                        exclude_val=exclude_val,
                        reason_seg_data=reason_seg_data,
                        explanatory=explanatory,
                        no_sampling=self.no_sampling,
                        dynamic_image_size=dynamic_image_size,
                        group_by_length=group_by_length,
                        use_packed_ds=use_packed_ds,
                    )
                )
            if self.no_sampling:
                assert len(self.all_datasets) == 1, "Only one dataset is allowed with the no-sampling strategy."

    def __len__(self):
        if self.no_sampling:
            return len(self.all_datasets[0])
        else:
            return self.samples_per_epoch

    def _enable_worker_distributed(self):
        if self.distributed_mode and not self.worker_distributed and self.worker_id is not None:
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id :: self.num_workers]
            logger.info(f"worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}")

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.no_sampling:
            data = self.all_datasets[0]
            result = data[i]
            return result
        else:
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
            data = self.all_datasets[ind]
            result = data[0]
            return result

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]["current_idx"]

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(f"[{self.ds_name}] [Worker id {self.worker_id}] begin to iter with {start_idx=}")

        for i in range(start_idx, len(self)):
            yield self[i]


class ValDataset(Dataset):
    def __init__(self, base_image_dir, tokenizer, val_dataset, image_size=448, data_type="reason_seg"):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg"))
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"
            if ds == "grefcoco":
                refer_api = G_REFER(os.path.join(self.base_image_dir, "refer_seg"), ds, splitBy)
            elif ds == "refzom":
                refer_api = REFZOM_REFER(os.path.join(self.base_image_dir, "refer_seg"), ds)
            else:
                refer_api = REFER(os.path.join(self.base_image_dir, "refer_seg"), ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(base_image_dir, "refer_seg", "images/saiapr_tc-12", item["file_name"])
                else:
                    item["file_name"] = os.path.join(base_image_dir, "refer_seg", "images/mscoco/images/train2014", item["file_name"])
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val
            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [ref]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = build_transform(is_train=False, input_size=image_size)

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        return Image.open(image_path).convert("RGB")

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=False, input_size=self.image_size, pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            image_info = images[idx]
            image_size = image_info["width"], image_info["height"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])
            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = Image.open(image_path).convert("RGB")
            images = [image]
            pixel_values = [self.transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)

            is_sentence = False
        else:
            # 读取 Image
            image_path = self.images[idx]
            image = Image.open(image_path).convert("RGB")
            images = [image]
            pixel_values = [self.transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)

            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        bboxes = []
        masks = []
        # 读取mask标签
        if self.data_type == "refer_seg":
            for i, ann_id in enumerate(sampled_ann_ids):
                # grefcoco multiple annid start
                if self.ds in ["grefcoco", "refzom"]:
                    no_target = ann_id == [-1] if self.ds == "grefcoco" else ann_id == []
                    if no_target:  # no target
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    elif len(ann_id) > 1:  # multi target / already merged ?
                        m = []
                        for sub_ann_id in ann_id:
                            sub_mask_info = annotations[sub_ann_id]["segmentation"]
                            if len(sub_mask_info) == 0:
                                sub_m = np.zeros((image_info["height"], image_info["width"], 1))
                            else:
                                if isinstance(sub_mask_info, dict):
                                    if isinstance(sub_mask_info["counts"], list):
                                        # convert to compressed RLE
                                        rle = mask.frPyObjects(sub_mask_info, image_info["height"], image_info["width"])
                                else:
                                    # filter out invalid polygons (< 3 points)
                                    polygons = [poly for poly in sub_mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                                    if len(polygons) == 0:
                                        continue  # ignore this instance
                                    rle = mask.frPyObjects(polygons, image_info["height"], image_info["width"])
                                sub_m = mask.decode(rle)
                                if sub_m.ndim < 3:
                                    assert sub_m.ndim == 2
                                    sub_m = sub_m[..., np.newaxis]
                            sub_m = np.sum(sub_m, axis=2)
                            m.append(sub_m)
                        m = np.sum(m, axis=0)[..., np.newaxis]
                    else:
                        assert len(ann_id) == 1 and ann_id[0] != -1
                        mask_info = annotations[ann_id[0]]["segmentation"]
                        if len(mask_info) == 0:
                            m = np.zeros((image_info["height"], image_info["width"], 1))
                        else:
                            if isinstance(mask_info, dict):
                                if isinstance(mask_info["counts"], list):
                                    # convert to compressed RLE
                                    rle = mask.frPyObjects(mask_info, image_info["height"], image_info["width"])
                            else:
                                # filter out invalid polygons (< 3 points)
                                polygons = [poly for poly in mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                                if len(polygons) == 0:
                                    continue  # ignore this instance
                                rle = mask.frPyObjects(polygons, image_info["height"], image_info["width"])
                            m = mask.decode(rle)
                            if m.ndim < 3:
                                assert m.ndim == 2
                                m = m[..., np.newaxis]
                    m = np.sum(m, axis=2)
                    masks.append(m)
                else:
                    ann = annotations[ann_id]
                    if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    else:
                        if type(ann["segmentation"][0]) == list:  # polygon
                            rle = mask.frPyObjects(
                                ann["segmentation"],
                                image_info["height"],
                                image_info["width"],
                            )
                        else:
                            rle = ann["segmentation"]
                            for i in range(len(rle)):
                                if not isinstance(rle[i]["counts"], bytes):
                                    rle[i]["counts"] = rle[i]["counts"].encode()
                        m = mask.decode(rle)
                    m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                    m = m.astype(np.uint8)  # convert to np.uint8
                    masks.append(m)
                    bboxes.append(ann["bbox"])
        else:
            masks = [mask_json]

        conversations = []

        # 处理conversation
        if self.data_type == "refer_seg":
            stripped_refers = [s.strip().strip(".") for s in sampled_sents]
            questions = DEFAULT_IMAGE_TOKEN + "\n What are " + ", ".join(stripped_refers) + "in this image? Please output segmentation masks."
            ref_w_segs = []
            assert len(stripped_refers) == len(masks)
            for t_idx, t in enumerate(stripped_refers):
                formatted_text = f"{t}:[REJ]" if masks[t_idx].sum() < 1.0 else f"{t}:[SEG]"
                ref_w_segs.append(formatted_text)
            answers = "Sure," + ", ".join(ref_w_segs) + "."
        else:
            i = 0
            while i < len(sampled_sents):
                text = sampled_sents[i].strip()
                seg_token = "[SEG]"
                if is_sentence:
                    questions = DEFAULT_IMAGE_TOKEN + "\n {} Please output segmentation mask.".format(text)
                    answers = seg_token
                else:
                    questions = DEFAULT_IMAGE_TOKEN + "\n What is {} in this image? Please output segmentation mask.".format(text)
                    answers = seg_token
                i += 1
        conversations = [{"from": "human", "value": questions}, {"from": "gpt", "value": answers}]

        if masks:
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks.astype(np.uint8))
            masks = masks.bool().byte()
        else:
            masks = None
        if bboxes:
            bboxes = np.stack(bboxes, axis=0)
        else:
            bboxes = None
        return image_path, pixel_values, conversations, masks, bboxes


class ValVisDataset(Dataset):
    def __init__(self, base_image_dir, tokenizer, val_dataset, image_size=448, data_type="reason_seg"):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg"))
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"
            if ds == "grefcoco":
                refer_api = G_REFER(os.path.join(self.base_image_dir, "refer_seg"), ds, splitBy)
            elif ds == "refzom":
                refer_api = REFZOM_REFER(os.path.join(self.base_image_dir, "refer_seg"), ds)
            else:
                refer_api = REFER(os.path.join(self.base_image_dir, "refer_seg"), ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(base_image_dir, "refer_seg", "images/saiapr_tc-12", item["file_name"])
                else:
                    item["file_name"] = os.path.join(base_image_dir, "refer_seg", "images/mscoco/images/train2014", item["file_name"])
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val
            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [ref]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = build_transform(is_train=False, input_size=image_size)

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        return Image.open(image_path).convert("RGB")

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=False, input_size=self.image_size, pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            image_info = images[idx]
            image_size = image_info["width"], image_info["height"]

            refs = img2refs[image_id]
            refs = refs[:1]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"][:1]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])
            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = Image.open(image_path).convert("RGB")
            images = [image]
            pixel_values = [self.transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)

            is_sentence = False
        else:
            # 读取 Image
            image_path = self.images[idx]
            image = Image.open(image_path).convert("RGB")
            images = [image]
            pixel_values = [self.transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)

            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        bboxes = []
        masks = []
        # 读取mask标签
        if self.data_type == "refer_seg":
            for i, ann_id in enumerate(sampled_ann_ids):
                # grefcoco multiple annid start
                if self.ds in ["grefcoco", "refzom"]:
                    no_target = ann_id == [-1] if self.ds == "grefcoco" else ann_id == []
                    if no_target:  # no target
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    elif len(ann_id) > 1:  # multi target / already merged ?
                        m = []
                        for sub_ann_id in ann_id:
                            sub_mask_info = annotations[sub_ann_id]["segmentation"]
                            if len(sub_mask_info) == 0:
                                sub_m = np.zeros((image_info["height"], image_info["width"], 1))
                            else:
                                if isinstance(sub_mask_info, dict):
                                    if isinstance(sub_mask_info["counts"], list):
                                        # convert to compressed RLE
                                        rle = mask.frPyObjects(sub_mask_info, image_info["height"], image_info["width"])
                                else:
                                    # filter out invalid polygons (< 3 points)
                                    polygons = [poly for poly in sub_mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                                    if len(polygons) == 0:
                                        continue  # ignore this instance
                                    rle = mask.frPyObjects(polygons, image_info["height"], image_info["width"])
                                sub_m = mask.decode(rle)
                                if sub_m.ndim < 3:
                                    assert sub_m.ndim == 2
                                    sub_m = sub_m[..., np.newaxis]
                            sub_m = np.sum(sub_m, axis=2)
                            m.append(sub_m)
                        m = np.sum(m, axis=0)[..., np.newaxis]
                    else:
                        assert len(ann_id) == 1 and ann_id[0] != -1
                        mask_info = annotations[ann_id[0]]["segmentation"]
                        if len(mask_info) == 0:
                            m = np.zeros((image_info["height"], image_info["width"], 1))
                        else:
                            if isinstance(mask_info, dict):
                                if isinstance(mask_info["counts"], list):
                                    # convert to compressed RLE
                                    rle = mask.frPyObjects(mask_info, image_info["height"], image_info["width"])
                            else:
                                # filter out invalid polygons (< 3 points)
                                polygons = [poly for poly in mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                                if len(polygons) == 0:
                                    continue  # ignore this instance
                                rle = mask.frPyObjects(polygons, image_info["height"], image_info["width"])
                            m = mask.decode(rle)
                            if m.ndim < 3:
                                assert m.ndim == 2
                                m = m[..., np.newaxis]
                    m = np.sum(m, axis=2)
                    masks.append(m)
                else:
                    ann = annotations[ann_id]
                    if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    else:
                        if type(ann["segmentation"][0]) == list:  # polygon
                            rle = mask.frPyObjects(
                                ann["segmentation"],
                                image_info["height"],
                                image_info["width"],
                            )
                        else:
                            rle = ann["segmentation"]
                            for i in range(len(rle)):
                                if not isinstance(rle[i]["counts"], bytes):
                                    rle[i]["counts"] = rle[i]["counts"].encode()
                        m = mask.decode(rle)
                    m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                    m = m.astype(np.uint8)  # convert to np.uint8
                    masks.append(m)
                    bboxes.append(ann["bbox"])
        else:
            masks = [mask_json]

        conversations = []

        # 处理conversation
        if self.data_type == "refer_seg":
            stripped_refers = [s.strip().strip(".") for s in sampled_sents]
            questions = DEFAULT_IMAGE_TOKEN + "\n What are " + ", ".join(stripped_refers) + "in this image? Please output segmentation masks."
            ref_w_segs = []
            assert len(stripped_refers) == len(masks)
            for t_idx, t in enumerate(stripped_refers):
                formatted_text = f"{t}:[REJ]" if masks[t_idx].sum() < 1.0 else f"{t}:[SEG]"
                ref_w_segs.append(formatted_text)
            answers = "Sure," + ", ".join(ref_w_segs) + "."
        else:
            i = 0
            while i < len(sampled_sents):
                text = sampled_sents[i].strip()
                seg_token = "[SEG]"
                if is_sentence:
                    questions = DEFAULT_IMAGE_TOKEN + "\n {} Please output segmentation mask.".format(text)
                    answers = seg_token
                else:
                    questions = DEFAULT_IMAGE_TOKEN + "\n What is {} in this image? Please output segmentation mask.".format(text)
                    answers = seg_token
                i += 1
        conversations = [{"from": "human", "value": questions}, {"from": "gpt", "value": answers}]

        if masks:
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks.astype(np.uint8))
            masks = masks.bool().byte()
        else:
            masks = None
        if bboxes:
            bboxes = np.stack(bboxes, axis=0)
        else:
            bboxes = None
        return image_path, pixel_values, conversations, masks, bboxes


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    dataset,
    sem_seg_data,
    refer_seg_data,
    refer_det_data,
    vqa_data,
    reason_seg_data,
    no_sampling=False,
    base_image_dir="",
    samples_per_epoch=0,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type="imagenet",
):
    datasets = []
    lengths = []
    # data_rank = dist.get_rank()
    # data_world_size = dist.get_world_size()
    data_rank = 0
    data_world_size = 0
    dataset = LazySupervisedDataset(
        data_args.conv_style,
        tokenizer,
        tcs_loader,
        num_image_token=model.num_image_token,
        image_size=data_args.force_image_size,
        pad2square=data_args.pad2square,
        group_by_length=group_by_length and not data_args.use_packed_ds,
        dynamic_image_size=dynamic_image_size,
        use_thumbnail=use_thumbnail,
        min_dynamic_patch=min_dynamic_patch,
        max_dynamic_patch=1,
        min_num_frame=min_num_frame,
        max_num_frame=max_num_frame,
        repeat_time=1,
        normalize_type=normalize_type,
        # hyperparameters for packed training
        use_packed_ds=data_args.use_packed_ds,
        data_rank=data_rank,
        data_world_size=data_world_size,
        distributed_mode=data_args.use_packed_ds,
        force_shuffle=data_args.use_packed_ds,
        random_seed=0,
        samples_per_epoch=samples_per_epoch,
        dataset=dataset,
        sem_seg_data=sem_seg_data,
        refer_seg_data=refer_seg_data,
        refer_det_data=refer_det_data,
        vqa_data=vqa_data,
        reason_seg_data=reason_seg_data,
        no_sampling=no_sampling,
        base_image_dir=base_image_dir,
    )
    datasets.append(dataset)
    if data_args.use_data_resampling:
        lengths.append(math.sqrt(len(dataset)))
    else:
        lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
            samples_per_epoch=samples_per_epoch,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset
