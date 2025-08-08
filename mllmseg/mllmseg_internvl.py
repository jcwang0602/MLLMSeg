from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from utils.common import conv
from utils.losses import dice_loss_plvl, sigmoid_focal_loss, mask_iou
from mllmseg.decoder import TransformerDecoder

logger = logging.get_logger(__name__)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            in_channels = in_channels // scale**2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale**2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).view(B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)


class RESDecoder(nn.Module):
    def __init__(self, txt_dim):
        super(RESDecoder, self).__init__()
        self.img_llm_down = nn.Linear(txt_dim, 1024)
        self.text_down = nn.Linear(txt_dim, 1024)
        self.text_out_adapter = nn.Linear(1024 * 3, 1024)
        decoder = dict(
            num_layers=1,
            layer=dict(
                d_model=1024,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
            ),
        )
        self.ca_layer_v2l = TransformerDecoder(decoder.copy())
        self.upsample_adapter_l = DySample(1024, scale=2)
        self.ca_layer_vl = TransformerDecoder(decoder.copy())

        self.conv1 = conv(1024, 196, freeze_bn=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 5, padding=2)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_pix, img_vit, img_llm, text_src):
        # img_pix: torch.Size([5, 3, 448, 448])
        # img_vit: torch.Size([5, 1024, 1024])
        # img_llm: torch.Size([5, 256, 4096])
        # text_src: torch.Size([5, 1, 4096])
        # print(f"img_pix.shape: {img_pix.shape}")
        # print(f"img_vit.shape: {img_vit.shape}")
        # print(f"img_llm.shape: {img_llm.shape}")
        # print(f"text_src.shape: {text_src.shape}")
        # exit()
        n_query = text_src.shape[0]
        if n_query == 0:
            return torch.empty(0, 1, 448, 448), torch.empty(0, 1, 448, 448), torch.empty(0, 1, 448, 448)
        text_src = self.text_down(text_src)
        # 处理形状
        img_llm_16 = self.img_llm_down(img_llm)
        img_vit_l_32 = self.ca_layer_v2l(img_vit.permute(1, 0, 2), img_llm_16.permute(1, 0, 2)).permute(1, 0, 2).view(n_query, 32, 32, 1024).permute(0, 3, 1, 2)
        vit_img_32 = img_vit.view(n_query, 32, 32, 1024).permute(0, 3, 1, 2)
        img_llm_32 = self.upsample_adapter_l(img_llm_16.view(n_query, 16, 16, 1024).permute(0, 3, 1, 2))
        out_img = torch.cat(
            [
                img_vit_l_32.flatten(2),
                vit_img_32.flatten(2),
                img_llm_32.flatten(2),
            ],
            dim=1,
        )
        out_img = self.text_out_adapter(out_img.permute(0, 2, 1))
        # 点乘
        tgt = self.ca_layer_vl(tgt=out_img.permute(1, 0, 2), memory=text_src.permute(1, 0, 2))
        tgt = tgt.permute(1, 2, 0).reshape(n_query, 1024, 32, 32).contiguous()
        x = self.conv1(tgt)
        x = x.permute(0, 2, 3, 1).reshape(-1, 32, 32, 14, 14)
        x = x.permute(0, 1, 3, 2, 4).reshape(-1, 1, 448, 448).contiguous()
        score_map_seg = self.conv2(x).sigmoid()
        return score_map_seg, img_vit_l_32, tgt


class MLLMSeg(InternVLChatModel):
    def __init__(
        self,
        config,
        tokenizer=None,
        data_type=torch.float16,
        init_decoder=False,
    ):
        super().__init__(config)
        self.seg_token_idx = None
        self.rej_token_idx = None
        self.img_context_token_idx = None
        # Loss 权重设置
        self.ce_loss_weight = None
        self.focal_loss_weight = None
        self.dice_loss_weight = None
        self.llm_config = config
        # TODO: for debug
        self.tokenizer = tokenizer
        self.data_type = data_type
        if init_decoder:
            self.init_mask_decoder()

    def init_mask_decoder(self):
        self.res_decoder = RESDecoder(txt_dim=self.llm_config.hidden_size).to(dtype=self.data_type)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
        gt_masks: torch.FloatTensor = None,
        gt_bboxes: torch.FloatTensor = None,
        resize_sam: torch.FloatTensor = None,
        image_sam: torch.FloatTensor = None,
        masks_sam: torch.FloatTensor = None,
        label_sam: torch.FloatTensor = None,
        offset: torch.LongTensor = None,
        conversations: Optional[List] = None,
        do_seg: torch.FloatTensor = None,
        reeval: bool = False,
        inference: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        batch_size = pixel_values.shape[0]
        device, dtype = pixel_values.device, pixel_values.dtype
        pixel_values = pixel_values.to(dtype=self.data_type)

        if inference:  # Segmentation Eval
            n_batch = 1
            length = input_ids.shape[0]
            assert pixel_values.shape[0] == 1
            pixel_values_extend = pixel_values.expand(length, -1, -1, -1).contiguous()  # for MLLM ViT
            output_hidden_states = []
            output_ids = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i, vit_embeds_internvit = super().forward(
                    pixel_values=pixel_values_extend[: end_i - start_i],
                    input_ids=input_ids[start_i:end_i],
                    attention_mask=attention_mask[start_i:end_i],
                    image_flags=image_flags[start_i:end_i],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                    statistics=statistics,
                    loss_weight=loss_weight,
                    loss_reduction_all_gather=loss_reduction_all_gather,
                    return_vit_feature=True,
                )
                output_hidden_states.append(output_i.hidden_states[-1])
                for k in range(length):
                    pred_output_ids = output_i.logits[k].argmax(dim=1)
                    pred_ids = input_ids[k].clone()
                    # [SEG] token prediction:
                    seg_rej_index_gt = ((pred_ids == self.seg_token_idx) | (pred_ids == self.rej_token_idx)).nonzero(as_tuple=True)[0]
                    seg_rej_index_pred = seg_rej_index_gt - 1
                    pred_seg_rej_values = torch.where(
                        (pred_output_ids[seg_rej_index_pred] == self.rej_token_idx),
                        self.rej_token_idx,
                        self.seg_token_idx,
                    )
                    pred_ids[seg_rej_index_gt] = pred_seg_rej_values.int()
                    output_ids.append(pred_ids)
                if reeval:
                    # Replace all [REJ] to [SEG], then re-eval
                    input_ids[input_ids == self.rej_token_idx] = self.seg_token_idx
                    output_i_reeval, vit_embeds_internvit = super().forward(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        image_flags=image_flags,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=True,
                        return_dict=return_dict,
                        statistics=statistics,
                        loss_weight=loss_weight,
                        loss_reduction_all_gather=loss_reduction_all_gather,
                        return_vit_feature=True,
                    )
                    output_hidden_states[-1] = output_i_reeval.hidden_states[-1]
                    torch.cuda.empty_cache()
            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
        else:
            pixel_values_list = []
            image_flags_list = []
            for i in range(len(offset) - 1):  # offset marks each begin and end index for each images.
                start_i, end_i = offset[i], offset[i + 1]
                pixel_values_i = pixel_values[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
                image_flags_i = image_flags[i].unsqueeze(0).expand(end_i - start_i, -1).contiguous()
                pixel_values_list.append(pixel_values_i)
                image_flags_list.append(image_flags_i)
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_flags = torch.cat(image_flags_list, dim=0)
            output, vit_embeds_internvit = super().forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                statistics=statistics,
                loss_weight=loss_weight,
                loss_reduction_all_gather=loss_reduction_all_gather,
                return_vit_feature=True,
            )
            output_hidden_states = output.hidden_states
        hidden_states = []
        hidden_states.append(output_hidden_states[-1])
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        rej_token_mask = input_ids[:, 1:] == self.rej_token_idx
        image_token_mask = input_ids[:, 1:] == self.img_context_token_id
        # 获取所有的 SEG Token
        last_hidden_state = last_hidden_state[:, :-1, :]
        mask_list_comp = []

        for lang_i in range(len(input_ids)):
            this_seg_token_m = seg_token_mask[lang_i].long() * 2
            this_rej_token_m = rej_token_mask[lang_i].long() * 1
            this_seg_rej = this_seg_token_m + this_rej_token_m
            gathered_idx = this_seg_rej.nonzero(as_tuple=True)[0]
            this_seg_rej = this_seg_rej[gathered_idx].eq(2).nonzero(as_tuple=True)[0]
            mask_list_comp.append(this_seg_rej)

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.tensor([0], dtype=torch.int64, device=device), seg_token_offset],
            dim=0,
        )

        # 将不同的SEG Token 分割成列表
        pred_embeddings_ = []
        num_pred_embs = len(seg_token_offset) - 1
        for i in range(num_pred_embs):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_  # [[s,s,s], [s,s]]

        # 根据预测的 SEG Token 进行解码
        # 获取LLM输出的IMG特征
        pred_masks = []
        tat_embeds = []
        img_vit_embeds = []
        img_llm_embeds = []
        img_cross_embeds_list = []
        for i in range(len(pred_embeddings)):
            text_embeds = pred_embeddings[i].unsqueeze(1)
            img_pix = pixel_values[i].unsqueeze(0).repeat(pred_embeddings[i].shape[0], 1, 1, 1)
            img_vit = vit_embeds_internvit[i].unsqueeze(0).repeat(pred_embeddings[i].shape[0], 1, 1)
            llm_img = last_hidden_state[i][image_token_mask[i]].unsqueeze(0).repeat(pred_embeddings[i].shape[0], 1, 1)
            # print(f"text_embeds: {text_embeds.shape}")
            # print(f"input_ids: {self.tokenizer.decode(input_ids[i])}")
            pred_mask, img_cross_embeds, tat = self.res_decoder(
                img_pix=img_pix,
                img_vit=img_vit,
                img_llm=llm_img,
                text_src=text_embeds,
            )
            pred_masks.append(pred_mask.squeeze(1))
            # img_vit_embeds.append(img_vit.reshape(text_embeds.shape[0], 32, 32, -1).permute(0, 3, 1, 2))
            # img_llm_embeds.append(llm_img.reshape(text_embeds.shape[0], 16, 16, -1).permute(0, 3, 1, 2))
            # img_cross_embeds_list.append(img_cross_embeds)
            # tat_embeds.append(tat)
        model_output = output
        if not inference:  # training, only train the [SEG] masks, ignore [REJ] masks
            pred_masks_, mask_list_comp_ = [], []
            # print(f"offset: {offset}")
            for k in range(len(offset) - 1):
                begin, end = offset[k], offset[k + 1]
                select_preds = pred_masks[begin:end]
                select_comps = mask_list_comp[begin:end]
                # print(f"select_preds len: {len(select_preds)}")
                # print(f"select_comps len: {len(select_comps)}")
                if len(select_preds) == 1:
                    pred_masks_.extend(select_preds)
                else:
                    pred_masks_.append(torch.cat(select_preds, dim=0))
                if len(select_comps) == 1:
                    mask_list_comp_.extend(select_comps)
                else:
                    mask_list_comp_.append(select_comps)
            pred_masks = pred_masks_
            mask_list_comp = mask_list_comp_
            assert len(gt_masks) == len(pred_masks)
            assert len(gt_masks) == len(mask_list_comp)
            pred_masks_ = []
            for b_idx in range(batch_size):
                L, h, w = pred_masks[b_idx].shape
                if L == 0:
                    pred_masks_.append(pred_masks[b_idx])
                    continue
                this_pred_masks_ = torch.zeros_like(gt_masks[b_idx], dtype=self.data_type)
                # print("*" * 100)
                # print(f"this_pred_masks_.shape: {this_pred_masks_.shape}", flush=True)
                # print(f"pred_masks[b_idx].shape: {pred_masks[b_idx].shape}", flush=True)
                # [5, 640, 640]
                if isinstance(mask_list_comp[b_idx], torch.Tensor):
                    this_pred_masks_[mask_list_comp[b_idx]] = F.interpolate(
                        pred_masks[b_idx].unsqueeze(1),
                        this_pred_masks_[mask_list_comp[b_idx]].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

                else:
                    assert isinstance(mask_list_comp[b_idx], list) and len(mask_list_comp[b_idx]) == L
                    for j in range(L):
                        this_pred_masks_[j] = pred_masks[b_idx][j : j + 1][mask_list_comp[b_idx][j]]
                pred_masks_.append(this_pred_masks_)
            pred_masks = pred_masks_
            for b in range(batch_size):
                for pm, gm in zip(pred_masks[b], gt_masks[b]):
                    assert pm.shape == gm.shape, f"b_idx: {b}, pm.shape: {pm.shape}, gm.shape: {gm.shape}"
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "output_ids": output_ids,
                "img_vit_embeds": img_vit_embeds,
                "img_llm_embeds": img_llm_embeds,
                "img_cross_embeds": img_cross_embeds_list,
                "tat_embeds": tat_embeds,
            }
        ce_loss = model_output.loss
        loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        mask_focal_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        num_masks = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        iou_total = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        for batch_idx in range(len(pred_masks)):
            if batch_idx >= len(gt_masks):
                raise ValueError(f"gt_masks are not in good shape with b_idx={batch_idx} >= len(gt_masks)={len(gt_masks)}, also len(preds)={len(pred_masks)}.")
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            if gt_mask.shape[0] == 0:
                continue
            if gt_mask.shape[0] != pred_mask.shape[0]:
                # i0, i1 = input_ids[0], input_ids[1]
                # i0, i1 = i0[i0 != IMAGE_TOKEN_INDEX], i1[i1 != IMAGE_TOKEN_INDEX]
                # print(f"gt: {gt_mask.shape}, pred: {pred_mask.shape}\n" + f"Prompt0: {self.llm_tokenizer.decode(i0)}\n" + f"Prompt1: {self.llm_tokenizer.decode(i1)}\n" + f"GT_MASK sum :{gt_mask.sum(dim=(1, 2))}\n")
                raise RuntimeError("Found it!")
            iou_total += mask_iou(pred_mask, gt_mask) * gt_mask.shape[0]
            mask_focal_loss += sigmoid_focal_loss(pred_mask, gt_mask) * gt_mask.shape[0]
            mask_dice_loss += dice_loss_plvl(pred_mask, gt_mask) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]
        mask_focal_loss = mask_focal_loss / (num_masks + 1e-8)
        mask_dice_loss = mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_focal_loss * self.focal_loss_weight + mask_dice_loss * self.dice_loss_weight
        loss = ce_loss * self.ce_loss_weight + mask_loss
        iou = iou_total / (num_masks + 1e-8)
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_focal_loss": mask_focal_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "iou": iou,
        }
