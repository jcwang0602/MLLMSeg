from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
from .decoder_layer.decoder_layer import TransformerDecoderLayer


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


class TransformerDecoderLayerWithPositionEmbedding(TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerDecoderLayerWithPositionEmbedding, self).__init__(*args, **kwargs)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        need_weights: bool = False,
        decoder_model=None,
    ):
        tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=need_weights)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(
            query=with_pos_embed(tgt, query_pos),
            key=with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        if decoder_model is not None:
            tgt2 = decoder_model.get_from_idx_vl(tgt2, decoder_model.retain_ind)
            tgt = decoder_model.get_from_idx_vl(tgt, decoder_model.retain_ind)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn


class TransformerDecoderWithPositionEmbedding(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm, is_eliminate=False):
        super(TransformerDecoderWithPositionEmbedding, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.is_eliminate = is_eliminate
        if self.is_eliminate:
            self.init_idx()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        if self.is_eliminate:
            origin_idx = torch.arange(0, output.shape[0], 1, dtype=torch.long, device=output.device)
            self.origin_idx = origin_idx.unsqueeze(0).expand(output.shape[1], -1).contiguous()
        self.tgt_key_padding_mask = tgt_key_padding_mask
        self.query_pos = query_pos
        features = []
        attns = []
        for layer_num, layer in enumerate(self.layers):
            output, attn = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=self.tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=self.query_pos,
                need_weights=False,
                decoder_model=self if self.is_eliminate else None,
            )
            features.append(output)
            attns.append(attn)
        output = self.norm(output)
        return output, features, attns

    def eliminate_token_by_norm(self, attn_weight):
        attn_weight = F.softmax(attn_weight, dim=1)
        attn_weight = self.process_attn_weight_by_norm(attn_weight)
        min_num, max_num = attn_weight.min(-1)[0].unsqueeze(-1), attn_weight.max(-1)[0].unsqueeze(-1)
        attn_weight = (attn_weight - min_num) / (max_num - min_num)
        rm_ind = attn_weight < 0.025
        random_tensor = torch.rand(rm_ind.shape, device=rm_ind.device)
        return rm_ind

    def process_attn_weight_by_norm(self, attn_weight):
        idx = self.idx.expand(attn_weight.shape[0], -1)
        idx_above = self.idx_above.expand(attn_weight.shape[0], -1)
        idx_below = self.idx_below.expand(attn_weight.shape[0], -1)
        idx_left = self.idx_left.expand(attn_weight.shape[0], -1)
        idx_right = self.idx_right.expand(attn_weight.shape[0], -1)
        idx_left_above = self.idx_left_above.expand(attn_weight.shape[0], -1)
        idx_left_below = self.idx_left_below.expand(attn_weight.shape[0], -1)
        idx_right_above = self.idx_right_above.expand(attn_weight.shape[0], -1)
        idx_right_below = self.idx_right_below.expand(attn_weight.shape[0], -1)

        attn_weight = (
            attn_weight.gather(1, idx)
            + attn_weight.gather(1, idx_above)
            + attn_weight.gather(1, idx_below)
            + attn_weight.gather(1, idx_left)
            + attn_weight.gather(1, idx_right)
            + attn_weight.gather(1, idx_left_above)
            + attn_weight.gather(1, idx_left_below)
            + attn_weight.gather(1, idx_right_above)
            + attn_weight.gather(1, idx_right_below)
        ) / 9
        return attn_weight

    def eliminate_token(self, attn_weight, origin_idx, size=None):
        # attn_weight (B,768)
        attn_weight = self.process_attn_weight(attn_weight, origin_idx)  # [8, 784]
        a, idx1 = torch.sort(attn_weight, dim=1, descending=True)
        if size is None:
            min_num, max_num = attn_weight.min(-1)[0].unsqueeze(-1), attn_weight.max(-1)[0].unsqueeze(-1)
            attn_weight = (attn_weight - min_num) / (max_num - min_num)
            rm_ind = attn_weight < 0.015
            rm_num = rm_ind.sum(-1)
            size = rm_num.min()
        random_tensor = torch.rand((size,), device=idx1.device)
        random_retain = random_tensor >= 0.9 
        retain_idx = idx1[:, -size:][:, random_retain]
        idx1 = idx1[:, :-size]
        idx1 = torch.cat([idx1, retain_idx], dim=-1)
        idx1, _ = torch.sort(idx1, dim=1, descending=False)
        return idx1

    def process_attn_weight(self, attn_weight, cur_idx):
        min_num = attn_weight.min(-1)[0]
        is_return_shape = False
        if attn_weight.shape[1] != 28**2:
            is_return_shape = True
            attn_weight = self.pad_patch(attn_weight, cur_idx, min_num)
        idx = self.idx.expand(attn_weight.shape[0], -1)
        idx_above = self.idx_above.expand(attn_weight.shape[0], -1)
        idx_below = self.idx_below.expand(attn_weight.shape[0], -1)
        idx_left = self.idx_left.expand(attn_weight.shape[0], -1)
        idx_right = self.idx_right.expand(attn_weight.shape[0], -1)
        idx_left_above = self.idx_left_above.expand(attn_weight.shape[0], -1)
        idx_left_below = self.idx_left_below.expand(attn_weight.shape[0], -1)
        idx_right_above = self.idx_right_above.expand(attn_weight.shape[0], -1)
        idx_right_below = self.idx_right_below.expand(attn_weight.shape[0], -1)

        attn_weight = (
            attn_weight.gather(1, idx)
            + attn_weight.gather(1, idx_above)
            + attn_weight.gather(1, idx_below)
            + attn_weight.gather(1, idx_left)
            + attn_weight.gather(1, idx_right)
            + attn_weight.gather(1, idx_left_above)
            + attn_weight.gather(1, idx_left_below)
            + attn_weight.gather(1, idx_right_above)
            + attn_weight.gather(1, idx_right_below)
        ) / 9
        if is_return_shape:
            tmp_idx = cur_idx[:, 1:] - 1
            attn_weight = attn_weight.gather(1, tmp_idx)
        return attn_weight

    def pad_patch(self, attn_weight, cur_idx, min_num):
        cur_idx = cur_idx[:, 1:] - 1
        new_weight = min_num.clone().unsqueeze(-1).repeat(1, 28**2)
        new_weight.scatter_(1, cur_idx, attn_weight)
        return new_weight

    def get_from_idx(self, x, idx):
        if len(x.shape) == 3:
            reg_token = x[0, :, :].unsqueeze(0)
            vis_token = x.gather(dim=0, index=idx)
            return torch.cat([reg_token, vis_token], dim=0)
        elif len(x.shape) == 2:
            reg_token = x[:, 0].unsqueeze(-1)  # [8, 1]
            vis_token = x.gather(dim=1, index=idx)
            return torch.cat([reg_token, vis_token], dim=1)
        elif len(x.shape) == 4:
            reg_token = x[:, :, 0:1, :]
            vis_token = x.gather(dim=2, index=idx)
            return torch.cat([reg_token, vis_token], dim=2)
        else:
            raise NotImplementedError

    def get_from_idx_vl(self, x, idx):
        if len(x.shape) == 3:
            vis_token = x.gather(dim=0, index=idx)
            return vis_token
        elif len(x.shape) == 2:
            vis_token = x.gather(dim=1, index=idx)
            return vis_token
        elif len(x.shape) == 4:
            vis_token = x.gather(dim=2, index=idx)
            return vis_token
        else:
            raise NotImplementedError

    def get_origin_idx(self, unm_idx, src_idx, dst_idx, origin_shape):
        un_len, merge_len = unm_idx.shape[1], src_idx.shape[1]
        tot_len = un_len + merge_len
        origin_idx = torch.zeros(origin_shape[:-1]).to(unm_idx.device).long()
        dst_origin_ind = torch.arange(un_len, tot_len + un_len, 1).to(unm_idx.device)

        idx = torch.arange(1, tot_len * 2, 2)
        origin_idx[:, idx] = dst_origin_ind
        origin_unm_idx = (unm_idx * 2).squeeze(-1)
        unm_origin_ind = torch.arange(0, un_len, 1).expand_as(origin_unm_idx).to(unm_idx.device)
        origin_idx = origin_idx.scatter(dim=1, index=origin_unm_idx, src=unm_origin_ind)

        origin_src_idx = (src_idx * 2).squeeze(-1)
        dst_idx = dst_idx.squeeze(-1) + un_len
        origin_idx = origin_idx.scatter(dim=1, index=origin_src_idx, src=dst_idx)
        return origin_idx

    def init_idx(self):
        idx = torch.arange(0, 28**2).unsqueeze(0)
        idx_above = idx - 28
        idx_above[idx_above < 0] += 28
        idx_below = idx + 28
        idx_below[idx_below >= (28**2)] -= 28
        idx_left = idx - 1
        idx_left[(idx_left % 28) == (28 - 1)] += 1
        idx_right = idx + 1
        idx_right[(idx_right % 28) == 0] -= 1
        idx_left_above = idx - (28 + 1)
        idx_left_above[idx < 28] += 28 + 1
        idx_left_above[(idx % 28) == 0] += 28 + 1
        idx_left_above[:, 0] -= 28 + 1
        idx_left_below = idx + (28 - 1)
        idx_left_below[idx > 28 * (28 - 1)] -= 28 - 1
        idx_left_below[(idx % 28) == 0] -= 28 - 1
        idx_right_above = idx - (28 - 1)
        idx_right_above[idx < (28 - 1)] += 28 - 1
        idx_right_above[idx % 28 == (28 - 1)] += 28 - 1
        idx_right_below = idx + (28 + 1)
        idx_right_below[idx >= 28 * (28 - 1)] -= 28 + 1
        idx_right_below[idx % 28 == (28 - 1)] -= 28 + 1
        idx_right_below[:, (28**2 - 1)] += 28 + 1
        try:
            self.idx = idx.cuda()
            self.idx_above = idx_above.cuda()
            self.idx_below = idx_below.cuda()
            self.idx_left = idx_left.cuda()
            self.idx_right = idx_right.cuda()
            self.idx_left_above = idx_left_above.cuda()
            self.idx_left_below = idx_left_below.cuda()
            self.idx_right_above = idx_right_above.cuda()
            self.idx_right_below = idx_right_below.cuda()
        except:
            self.idx = idx
            self.idx_above = idx_above
            self.idx_below = idx_below
            self.idx_left = idx_left
            self.idx_right = idx_right
            self.idx_left_above = idx_left_above
            self.idx_left_below = idx_left_below
            self.idx_right_above = idx_right_above
            self.idx_right_below = idx_right_below


class TransformerDecoder(nn.Module):
    def __init__(self, decoder, patch_length=0, eliminated_threshold=0):
        super(TransformerDecoder, self).__init__()
        self.d_model = decoder["layer"]["d_model"]
        self.decoder = TransformerDecoderWithPositionEmbedding(TransformerDecoderLayerWithPositionEmbedding(**decoder.pop("layer")), **decoder, norm=nn.LayerNorm(self.d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        return_attn: bool = False,
    ):
        tgt, features, attns = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
        if return_attn:
            return tgt, features, attns
        else:
            return tgt
