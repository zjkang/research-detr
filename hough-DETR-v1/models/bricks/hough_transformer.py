import copy
import math
from typing import Tuple

import torch
import torchvision
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.hough import Hough
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import PositionEmbeddingLearned, get_sine_pos_embed

from util.misc import inverse_sigmoid


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


BN_MOMENTUM = 0.1

class HeatmapPredictor(nn.Module):
    def __init__(
            self,
            num_classes,
            embed_dim,
            region_num,
            vote_field_size=17,
            head_conv=64):
        super().__init__()
        self.inplanes = embed_dim # default=64
        self.region_num = region_num
        self.num_classes = num_classes
        self.vote_field_size = vote_field_size
        self.embed_dim = embed_dim
        self.head_conv = head_conv
        self.deconv_layers = self._make_deconv_layer2(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )
        # voting-map vs voting diff
        # voting-map is learnable, voting is aggregated result from Hough
        # learn to generate voting-map
        self.voting_map_hm = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.head_conv,
                kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.num_classes,
                kernel_size=1, stride=1, padding=0))
        self.voting_hm = Hough(
            region_num=self.region_num,
            vote_field_size=self.vote_field_size,
            num_classes=self.num_classes
        )

        self.init_weights()  # 在__init__的最后调用初始化方法

    def _make_deconv_layer2(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = nn.Conv2d(self.inplanes, planes,
                    kernel_size=3, stride=1,
                    padding=1, dilation=1, bias=False)
            self.fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def fill_fc_weights(layers):
        # layers.modules() 返回该模块中的所有子模块（如卷积层、全连接层、激活函数等）
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # x: (b, c=embed_dim, h, w)
    def forward(self, x):
        # x = (b, region_num * num_classes, h, w)
        # NOTE!!! does it need to add this?
        x = self.deconv_layers(x)
        # x = (b, region_num * num_classes, h, w)
        x = self.voting_map_hm(x)
        # out = (b, num_classes, h, w)
        out = self.voting_hm(x)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class HoughTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        level_filter_ratio: Tuple = (0.25, 0.5, 1.0, 1.0),
        layer_filter_ratio: Tuple = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
        region_num: int = 17,
        vote_field_size: int = 65
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes
        self.region_num = region_num
        self.vote_field_size = vote_field_size

        # salience parameters
        self.register_buffer("level_filter_ratio", torch.Tensor(level_filter_ratio))
        self.register_buffer("layer_filter_ratio", torch.Tensor(layer_filter_ratio))
        self.alpha = nn.Parameter(torch.Tensor(3), requires_grad=True)

        # model structure
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.encoder.enhance_mcsp = self.encoder_class_head
        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)
        self.enc_mask_voting_hm_predictor = HeatmapPredictor(
            self.num_classes, self.embed_dim, self.vote_field_size)

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        # initialize encoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        # initialize alpha
        self.alpha.data.uniform_(-0.3, 0.3)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        noised_label_query,
        noised_box_query,
        attn_mask,
    ):
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)

        backbone_output_memory = self.gen_encoder_output_proposals(
            feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes
        )[0]

        # calculate filtered tokens numbers for each feature map
        reverse_multi_level_masks = [~m for m in multi_level_masks]
        valid_token_nums = torch.stack([m.sum((1, 2)) for m in reverse_multi_level_masks], -1)
        focus_token_nums = (valid_token_nums * self.level_filter_ratio).int()
        level_token_nums = focus_token_nums.max(0)[0]
        focus_token_nums = focus_token_nums.sum(-1)

        # from high level to low level
        batch_size = feat_flatten.shape[0]
        selected_score = []
        selected_inds = []
        salience_score = []
        for level_idx in range(spatial_shapes.shape[0] - 1, -1, -1):
            start_index = level_start_index[level_idx]
            end_index = level_start_index[level_idx + 1] if level_idx < spatial_shapes.shape[0] - 1 else None
            level_memory = backbone_output_memory[:, start_index:end_index, :]
            mask = mask_flatten[:, start_index:end_index]
            # update the memory using the higher-level score_prediction
            if level_idx != spatial_shapes.shape[0] - 1:
                upsample_score = torch.nn.functional.interpolate(
                    score,
                    size=spatial_shapes[level_idx].unbind(),
                    mode="bilinear",
                    align_corners=True,
                )
                upsample_score = upsample_score.view(batch_size, -1, spatial_shapes[level_idx].prod())
                upsample_score = upsample_score.transpose(1, 2)
                level_memory = level_memory + level_memory * upsample_score * self.alpha[level_idx]
            # predict the foreground score of the current layer
            score = self.enc_mask_predictor(level_memory)
            valid_score = score.squeeze(-1).masked_fill(mask, score.min())
            score = score.transpose(1, 2).view(batch_size, -1, *spatial_shapes[level_idx])

            # get the topk salience index of the current feature map level
            level_score, level_inds = valid_score.topk(level_token_nums[level_idx], dim=1)
            level_inds = level_inds + level_start_index[level_idx]
            salience_score.append(score)
            selected_inds.append(level_inds)
            selected_score.append(level_score)

        selected_score = torch.cat(selected_score[::-1], 1)
        index = torch.sort(selected_score, dim=1, descending=True)[1]
        selected_inds = torch.cat(selected_inds[::-1], 1).gather(1, index)

        ## my code
        # Generate heatmaps for each level
        heatmaps = []
        for level_idx in range(spatial_shapes.shape[0] - 1, -1, -1):
            start_index = level_start_index[level_idx]
            end_index = level_start_index[level_idx + 1] if level_idx < spatial_shapes.shape[0] - 1 else None
            level_memory = backbone_output_memory[:, start_index:end_index, :]

            # Reshape level_memory to match spatial dimensions
            # level_memory: (b, h, w, channels)
            level_memory = level_memory.view(batch_size, *spatial_shapes[level_idx], -1)

            # Upsample to the first level's resolution
            if level_idx > 0:
                level_memory = torch.nn.functional.interpolate(
                    level_memory,
                    size=spatial_shapes[0].unbind(),
                    mode="bilinear",
                    align_corners=True
                )

            # Generate hm voting
            heat_map = self.enc_mask_voting_hm_predictor(level_memory)
            heatmaps.append(heat_map)

        # Combine heatmaps from all levels
        # combined_voting_hm: (b, num_classes, h, w)
        combined_voting_hm = torch.stack(heatmaps).sum(dim=0)
        # Do I need normalization?
        combined_voting_hm = combined_voting_hm / combined_voting_hm.max()
        ## my code

        # create layer-wise filtering
        num_inds = selected_inds.shape[1]
        # change dtype to avoid shape inference error during exporting ONNX
        cast_dtype = num_inds.dtype if torchvision._is_tracing() else torch.int64
        layer_filter_ratio = (num_inds * self.layer_filter_ratio).to(cast_dtype)
        selected_inds = [selected_inds[:, :r] for r in layer_filter_ratio]
        salience_score = salience_score[::-1]
        foreground_score = self.flatten_multi_level(salience_score).squeeze(-1)
        foreground_score = foreground_score.masked_fill(mask_flatten, foreground_score.min())

        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # salience input
            foreground_score=foreground_score,
            focus_token_nums=focus_token_nums,
            foreground_inds=selected_inds,
            multi_level_masks=multi_level_masks,
        )

        if self.neck is not None:
            feat_unflatten = memory.split(spatial_shapes.prod(-1).unbind(), dim=1)
            feat_unflatten = dict((
                i,
                feat.transpose(1, 2).contiguous().reshape(-1, self.embed_dim, *spatial_shape),
            ) for i, (feat, spatial_shape) in enumerate(zip(feat_unflatten, spatial_shapes)))
            feat_unflatten = list(self.neck(feat_unflatten).values())
            memory = torch.cat([feat.flatten(2).transpose(1, 2) for feat in feat_unflatten], dim=1)

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        if torchvision._is_tracing():
            topk = torch.min(torch.tensor(self.two_stage_num_proposals * 4), enc_outputs_class.shape[1])
        else:
            topk = min(self.two_stage_num_proposals * 4, enc_outputs_class.shape[1])
        topk_scores, topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)
        topk_index = self.nms_on_topk_index(
            topk_scores, topk_index, spatial_shapes, level_start_index, iou_threshold=0.3
        ).unsqueeze(-1)
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, self.num_classes))
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()
        target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        # decoder
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        return outputs_classes, outputs_coords,\
            enc_outputs_class, enc_outputs_coord,\
            salience_score, norm_voting_hm

    @staticmethod
    def fast_repeat_interleave(input, repeats):
        """torch.Tensor.repeat_interleave is slow for one-dimension input for unknown reasons. 
        This is a simple faster implementation. Notice the return shares memory with the input.

        :param input: input Tensor
        :param repeats: repeat numbers of each element in the specified dim
        :param dim: the dimension to repeat, defaults to None
        """
        # the following inplementation runs a little faster under one-dimension settings
        return torch.cat([aa.expand(bb) for aa, bb in zip(input, repeats)])

    @torch.no_grad()
    def nms_on_topk_index(
        self, topk_scores, topk_index, spatial_shapes, level_start_index, iou_threshold=0.3
    ):
        batch_size, num_topk = topk_scores.shape
        if torchvision._is_tracing():
            num_pixels = spatial_shapes.prod(-1).unbind()
        else:
            num_pixels = spatial_shapes.prod(-1).tolist()

        # flatten topk_scores and topk_index for batched_nms
        topk_scores, topk_index = map(lambda x: x.view(-1), (topk_scores, topk_index))

        # get level coordinates for queries and construct boxes for them
        level_index = torch.arange(level_start_index.shape[0], device=level_start_index.device)
        feat_width, start_index, level_idx = map(
            lambda x: self.fast_repeat_interleave(x, num_pixels)[topk_index],
            (spatial_shapes[:, 1], level_start_index, level_index),
        )
        topk_spatial_index = topk_index - start_index
        x = topk_spatial_index % feat_width
        y = torch.div(topk_spatial_index, feat_width, rounding_mode="trunc")
        coordinates = torch.stack([x - 1.0, y - 1.0, x + 1.0, y + 1.0], -1)

        # get unique idx for queries in different images and levels
        image_idx = torch.arange(batch_size).repeat_interleave(num_topk, 0)
        image_idx = image_idx.to(level_idx.device)
        idxs = level_idx + level_start_index.shape[0] * image_idx

        # perform batched_nms
        indices = torchvision.ops.batched_nms(coordinates, topk_scores, idxs, iou_threshold)

        # stack valid index
        results_index = []
        if torchvision._is_tracing():
            min_num = torch.tensor(self.two_stage_num_proposals)
        else:
            min_num = self.two_stage_num_proposals
        # get indices in each image
        for i in range(batch_size):
            topk_index_per_image = topk_index[indices[image_idx[indices] == i]]
            if torchvision._is_tracing():
                min_num = torch.min(topk_index_per_image.shape[0], min_num)
            else:
                min_num = min(topk_index_per_image.shape[0], min_num)
            results_index.append(topk_index_per_image)
        return torch.stack([index[:min_num] for index in results_index])


class HoughTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
        # focus parameter
        topk_sa=300,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.topk_sa = topk_sa

        # pre attention
        self.pre_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout, batch_first=True)
        self.pre_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.pre_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.pre_attention.out_proj.weight)
        # initilize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        value,  # focus parameter
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
        # focus parameter
        score_tgt=None,
        foreground_pre_layer=None,
    ):
        mc_score = score_tgt.max(-1)[0] * foreground_pre_layer
        select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]
        select_tgt_index = select_tgt_index.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        select_tgt = torch.gather(query, 1, select_tgt_index)
        select_pos = torch.gather(query_pos, 1, select_tgt_index)
        query_with_pos = key_with_pos = self.with_pos_embed(select_tgt, select_pos)
        tgt2 = self.pre_attention(
            query_with_pos,
            key_with_pos,
            select_tgt,
        )[0]
        select_tgt = select_tgt + self.pre_dropout(tgt2)
        select_tgt = self.pre_norm(select_tgt)
        query = query.scatter(1, select_tgt_index, select_tgt)

        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class HoughTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        # learnt background embed for prediction
        self.background_embedding = PositionEmbeddingLearned(200, num_pos_feats=self.embed_dim // 2)

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [n, s, l, 2]
        return reference_points

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        query_pos=None,
        query_key_padding_mask=None,
        # salience input
        foreground_score=None,
        focus_token_nums=None,
        foreground_inds=None,
        multi_level_masks=None,
    ):
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=query.device)
        b, n, s, p = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        value = output = query
        for layer_id, layer in enumerate(self.layers):
            inds_for_query = foreground_inds[layer_id].unsqueeze(-1).expand(-1, -1, self.embed_dim)
            query = torch.gather(output, 1, inds_for_query)
            query_pos = torch.gather(ori_pos, 1, inds_for_query)
            foreground_pre_layer = torch.gather(foreground_score, 1, foreground_inds[layer_id])
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)
            ).view(b, -1, s, p)
            score_tgt = self.enhance_mcsp(query)
            query = layer(
                query,
                query_pos,
                value,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
                score_tgt,
                foreground_pre_layer,
            )
            outputs = []
            for i in range(foreground_inds[layer_id].shape[0]):
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                query_no_pad = query[i][:focus_token_nums[i]]
                outputs.append(
                    output[i].scatter(
                        0,
                        foreground_inds_no_pad.unsqueeze(-1).repeat(1, query.size(-1)),
                        query_no_pad,
                    )
                )
            output = torch.stack(outputs)

        # add learnt embedding for background
        if multi_level_masks is not None:
            background_embedding = [
                self.background_embedding(mask).flatten(2).transpose(1, 2) for mask in multi_level_masks
            ]
            background_embedding = torch.cat(background_embedding, dim=1)
            background_embedding.scatter_(1, inds_for_query, 0)
            background_embedding *= (~query_key_padding_mask).unsqueeze(-1)
            output = output + background_embedding

        return output


class HoughTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        self_attn_mask=None,
        key_padding_mask=None,
    ):
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class HoughTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        self.class_head = nn.ModuleList([nn.Linear(self.embed_dim, num_classes) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([MLP(self.embed_dim, self.embed_dim, 4, 3) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
    ):
        outputs_classes = []
        outputs_coords = []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_mask,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query)) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # iterative bounding box refinement
            reference_points = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points.detach())
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords
