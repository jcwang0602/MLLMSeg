import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, Optional


import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from mllmseg_internvl.mllmseg import MLLMSeg
from internvl.patch import (
    concat_pad_vgdata_collator,
    replace_internlm2_attention_class,
    replace_llama_attention_class,
    replace_llama_rmsnorm_with_fused_rmsnorm,
    replace_phi3_attention_class,
    replace_qwen2_attention_class,
    replace_train_dataloader,
    replace_train_sampler,
)
from mllmseg_internvl.constants import (
    SEG_TOKEN,
    REJ_TOKEN,
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from mllmseg_internvl.dataset_packed import packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import enable_default_handler, enable_explicit_format, set_verbosity
from mllmseg_internvl.dataset_vg import build_datasets


# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="OpenGVLab/InternVL2_5-1B", metadata={"help": "Path to a pretrained model (local or from huggingface.co/models)."})
    llm_path: Optional[str] = field(default=None, metadata={"help": "Path to a pretrained model (local or from huggingface.co/models)."})
    mlp_path: Optional[str] = field(default=None, metadata={"help": "Path to a pretrained model (local or from huggingface.co/models)."})
    freeze_llm: bool = field(default=False, metadata={"help": "Set to True to freeze the LLM. Default is False."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Set to True to freeze the ViT. Default is False."})
    freeze_mlp: bool = field(default=False, metadata={"help": "Set to True to freeze the MLP. Default is False."})
    unfreeze_vit_layers: int = field(default=0, metadata={"help": "Specify the number of ViT layers to unfreeze. Default is 0."})
    vision_select_layer: int = field(default=-1, metadata={"help": "Specify the layer of ViT feature map to use. Default is -1 for the last layer."})
    use_backbone_lora: int = field(default=0, metadata={"help": "Set the LoRA adapter rank for the ViT. Default is 0."})
    use_llm_lora: int = field(default=0, metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 0."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "Set to True to unfreeze the head of LLM. Default is False."})
    grad_checkpoint: bool = field(default=True, metadata={"help": "Set to True to use gradient checkpointing. Default is True."})
    drop_path_rate: float = field(default=0.0, metadata={"help": "Set the drop path rate for the ViT. Default is 0."})
    ps_version: Literal["v1", "v2"] = field(default="v2", metadata={"help": "Specify the version of pixel shuffle implementation. Default is v2."})
    use_fast_tokenizer: bool = field(default=False, metadata={"help": "Set to True to use the fast mode of the tokenizer."})
    use_liger: bool = field(default=False, metadata={"help": "Set to True to use the liger kernel."})
    pretrained_model: Optional[str] = field(default=None, metadata={"help": "Path to a pretrained model (local or from huggingface.co/models)."})


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=8192, metadata={"help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")})
    force_image_size: int = field(default=448, metadata={"help": "Set the desired size for the image. Default is 448."})
    down_sample_ratio: float = field(default=0.5, metadata={"help": "Set the desired down-sampling ratio for the image. Default is 0.5."})
    pad2square: bool = field(default=False, metadata={"help": "Pad the image to a square shape if set to True. Default is False."})
    conv_style: str = field(default="internvl2_5", metadata={"help": "Prompt style for a conversation."})
    meta_path: str = field(default=None, metadata={"help": "The path of the meta file of datasets."})
    use_data_resampling: bool = field(default=False, metadata={"help": "Set to True to use data resampling. Default is False."})
    dynamic_image_size: bool = field(default=False, metadata={"help": "Set to True to use dynamic high resolution strategy. Default is False."})
    use_thumbnail: bool = field(default=True, metadata={"help": "Set to True to add a thumbnail image. Default is False."})
    min_dynamic_patch: int = field(default=1, metadata={"help": "The minimum number of dynamic patches. Default is 1."})
    max_dynamic_patch: int = field(default=6, metadata={"help": "The maximum number of dynamic patches. Default is 12."})
    min_num_frame: int = field(default=8, metadata={"help": "The minimum number of frames for video data. Default is 8."})
    max_num_frame: int = field(default=32, metadata={"help": "The maximum number of frames for video data. Default is 32."})
    normalize_type: Literal["imagenet", "clip", "siglip"] = field(default="imagenet", metadata={"help": "The normalization type for the image. Default is imagenet."})
    use_packed_ds: bool = field(default=False, metadata={"help": "Whether to use packed dataset for efficient training. Default is False."})
    num_images_expected: int = field(default=40, metadata={"help": "The maximum number of images per packed sample. Default is 40."})
    max_packed_tokens: int = field(default=8192, metadata={"help": "The required token length of per packed sample. Default is 8192."})
    max_buffer_size: int = field(default=20, metadata={"help": "The buffer size of the packed dataset. Default is 20."})
    log_freq: int = field(default=1000, metadata={"help": "The log frequency of the packed dataset. Default is 1000."})
    strict_mode: bool = field(default=True, metadata={"help": "Whether to pad the number of images to satisfy num_images_expected. Default is True."})
    replacement: bool = field(default=False, metadata={"help": "Whether to restart the dataset after it is exhausted. Default is False."})
    allow_overflow: bool = field(default=False, metadata={"help": "Whether to drop the sample over the specified max_packed_tokens. Default is False."})
    loss_reduction: str = field(default="token", metadata={"help": "Loss reduction method. Default is token."})
    loss_reduction_all_gather: bool = field(default=False, metadata={"help": "Whether to gather all during loss reduction. Default is False."})
    # Dataset
    train_dataset: str = field(default="", metadata={"help": "Traing Dataset"})
    sem_seg_data: str = field(default="", metadata={"help": "Traing Dataset"})
    refer_seg_data: str = field(default="", metadata={"help": "Traing Dataset"})
    refer_det_data: str = field(default="", metadata={"help": "Traing Dataset"})
    vqa_data: str = field(default="", metadata={"help": "Traing Dataset"})
    reason_seg_data: str = field(default="", metadata={"help": "Traing Dataset"})
    no_sampling: bool = field(default=False, metadata={"help": ""})
    base_image_dir: str = field(default="/share/wangjingchao/gres_datasets", metadata={"help": ""})
    total_steps: int = field(default=50000, metadata={"help": "Total steps"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    ce_loss_weight: float = field(default=1.0, metadata={"help": "Set the weight for the cross entropy loss. Default is 1.0."})
    focal_loss_weight: float = field(default=1.0, metadata={"help": "Set the weight for the focal loss. Default is 1.0."})
    dice_loss_weight: float = field(default=1.0, metadata={"help": "Set the weight for the dice loss. Default is 1.0."})


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


# 自定义 Trainer 并重写 compute_loss 方法
class MyTrainer(Trainer):
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None, tokenizer=None, model_init=None, compute_metrics=None, callbacks=None, optimizers=(None, None), preprocess_logits_for_metrics=None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # 记录额外的损失，例如通过日志或回调
        if "mask_loss" in outputs:
            self.log(
                {
                    "loss": loss.item(),
                    "ce_loss": round(outputs["ce_loss"].item(), 4),
                    "mask_loss": round(outputs["mask_loss"].item(), 4),
                    "mask_focal_loss": round(outputs["mask_focal_loss"].item(), 4),
                    "mask_dice_loss": round(outputs["mask_dice_loss"].item(), 4),
                    "iou": round(outputs["iou"].item(), 4),
                }
            )

        return (loss, outputs) if return_outputs else loss


def main():
    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))  # type: ignore
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.use_packed_ds = data_args.use_packed_ds

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # TODO:
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f"Loading Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=model_args.use_fast_tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        SEG_TOKEN,
        REJ_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = None

    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    logger.info("Loading InternVLChatModel...")
    config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    if config.llm_config.model_type == "internlm2":
        config.llm_config.attn_implementation = "flash_attention_2"  # for InternLM
        logger.info("Using flash_attention_2 for InternLM")
    else:
        config.llm_config._attn_implementation = "flash_attention_2"  # for LLaMA
        logger.info("Using flash_attention_2 for LLaMA")
    data_args.conv_style = config.template
    config.select_layer = model_args.vision_select_layer
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail = data_args.use_thumbnail
    config.ps_version = model_args.ps_version
    config.min_dynamic_patch = data_args.min_dynamic_patch
    config.max_dynamic_patch = data_args.max_dynamic_patch
    model = MLLMSeg.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        config=config,
        tokenizer=tokenizer,
        data_type=torch.bfloat16,
    )

    model.init_vg_model()
    model.img_context_token_id = img_context_token_id
    model.rej_token_idx = tokenizer.convert_tokens_to_ids("[REJ]")
    model.seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")

    model.ce_loss_weight = training_args.ce_loss_weight
    model.focal_loss_weight = training_args.focal_loss_weight
    model.dice_loss_weight = training_args.dice_loss_weight

    assert model.config.downsample_ratio == data_args.down_sample_ratio
    logger.info("Finished")

    patch_size = model.config.vision_config.patch_size
    logger.info(f"model.config.force_image_size: {model.config.force_image_size}")
    logger.info(f"data_args.force_image_size: {data_args.force_image_size}")
    logger.info(f"model.config.vision_config.image_size: {model.config.vision_config.image_size}")
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f"Resizing position embedding from {model.config.vision_config.image_size} to {data_args.force_image_size}...")
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size, new_size=data_args.force_image_size, patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio**2))
    logger.info(f"model.num_image_token: {model.num_image_token}")
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    if model_args.pretrained_model:
        state_dict = torch.load(os.path.join(model_args.pretrained_model, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        logger.info("Model Info: Pretrained Weight Load Success!!!")

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.requires_grad = True

    trainable_parts_keys = ["model.norm.weight", "output", "tok_embeddings"]
    for n, p in model.named_parameters():
        if any([x in n for x in trainable_parts_keys]):
            p.requires_grad_(True)
    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)
    total_params = sum(p.numel() for p in model.res_decoder.parameters())
    dec_params = sum(p.numel() for p in model.res_decoder.parameters())
    logger.info(f"模型信息：模型参数量为 {total_params}, decoder参数量为 {dec_params}")
    # 创建数据集
    steps = data_args.total_steps
    samples_per_epoch = steps * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count()
    logger.info(f"Dataset Info: steps: {steps}")
    logger.info(f"Dataset Info: per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    logger.info(f"Dataset Info: gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"Dataset Info: n_gpu: {torch.cuda.device_count()}")
    logger.info(f"Dataset Info: samples_per_epoch: {samples_per_epoch}")
    train_dataset = build_datasets(
        data_args,
        tokenizer,
        tcs_loader,
        model,
        dataset=data_args.train_dataset,
        sem_seg_data=data_args.sem_seg_data,
        refer_seg_data=data_args.refer_seg_data,
        refer_det_data=data_args.refer_det_data,
        vqa_data=data_args.vqa_data,
        reason_seg_data=data_args.reason_seg_data,
        no_sampling=data_args.no_sampling,
        base_image_dir=data_args.base_image_dir,
        samples_per_epoch=samples_per_epoch,
        group_by_length=False,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
    )

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_vgdata_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_vgdata_collator

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics["train_samples"] = len(train_dataset)
        except:
            metrics["train_samples"] = -1

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
