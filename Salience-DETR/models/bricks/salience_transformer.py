import copy
import math
from typing import Tuple

import torch
import torchvision
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import PositionEmbeddingLearned, get_sine_pos_embed
from util.misc import inverse_sigmoid


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        # 隐藏层的维度（h_dim > in_dim）
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
        # torch.split(z, self.h_dim // 2, dim=-1)：将输出 z 分为两部分，
        # 一部分是 z_local，另一部分是 z_global，每部分的维度为 h_dim // 2
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(
            dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        # torch.cat([z_local, z_global], dim=-1)：将局部信息 z_local 和全局信息 z_global 拼接起来，形成一个新的表示 z
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


class SalienceTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        # feature map 4 levels
        level_filter_ratio: Tuple = (0.25, 0.5, 1.0, 1.0),
        # encoder layer 6
        layer_filter_ratio: Tuple = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        # 两阶段检测器中生成的目标提议数量
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # salience parameters
        # 针对不同的特征层（level），即特征金字塔的每一层
        # 默认值：(0.25, 0.5, 1.0, 1.0)，这意味着：
        # 第1层只保留 25% 的显著性区域；
        # 第2层保留 50%；
        # 第3和第4层保留所有的显著性区域
        self.register_buffer("level_filter_ratio",
                             torch.Tensor(level_filter_ratio))
        # 不同的网络层（layer），即网络的深度层次
        # 最初的层保留所有的显著性区域；
        # 随着层次加深，保留的显著性特征比例逐步降低；
        # 到最深的层次，只保留 20% 的显著性区域
        self.register_buffer("layer_filter_ratio",
                             torch.Tensor(layer_filter_ratio))
        self.alpha = nn.Parameter(torch.Tensor(3), requires_grad=True)

        # model structure
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        # 这一行代码创建了一个嵌入层，为每个目标查询生成一个可学习的高维向量，
        # 帮助模型在特征图中找到目标并进行预测。这是目标检测任务中常用的策略
        # 嵌入是可学习的参数，会在训练过程中更新 (object queries)
        # 解码器的输入实际上是通过 tgt_embed 生成的目标查询
        #  target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)
        # 这个 target 才是真正传递给解码器的输入
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        # enhance_mcsp 中的 "mcsp" 很可能是 "Multi-Class Score Prediction",
        # 一个用于增强或改进某种多类别分数预测的函数或模块
        # enhance_mcsp 和 encoder_class_head 使用完全相同的参数
        self.encoder.enhance_mcsp = self.encoder_class_head
        #  its function in predicting masks or salience scores
        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        # 目标嵌入层 (tgt_embed) 的权重通过正态分布进行初始化
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

    # outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask = self.transformer(
    #     multi_level_feats,
    #     multi_level_masks,
    #     multi_level_position_embeddings,
    #     noised_label_query,
    #     noised_box_query,
    #     attn_mask=attn_mask,
    # )
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
        # feat_flatten.shape = (b, s, c), s=sum(h_i x w_i): total number of tokens
        # all the feature map tokens from all levels concatenated along the second dimension
        feat_flatten = self.flatten_multi_level(
            multi_level_feats)
        # mask_flatten.shape = (b, s)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        # lvl_pos_embed_flatten.shape = (b,s,c)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        # It's a tensor that contains the ratio of valid (non-padded) area for each feature map level 
        # in both width and height dimensions.
        # valid_ratios.shape = (b, l, 2), L is the number of feature levels
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(
            multi_level_masks)

        # backbone_output_memory.shape = (b, s, c=embed_dim)
        # gen_encoder_output_proposals purposes:
        # Components of a proposal
        # Center coordinates (x, y): Derived from a grid over the feature map
        # Width and height: Initially set to a small value that increases with feature level.
        # These proposals are not the final predictions
        # 4. Characteristics:
        # They cover the entire feature map uniformly.
        # The size of proposals increases for higher-level (more coarse) feature maps.
        # They are normalized to be in the range [0, 1] relative to the image size
        # 5. Role in the model:
        # These proposals are not the final predictions.
        # They serve as a starting point for the transformer decoder to refine.
        # The model will learn to adjust these proposals to better fit actual objects in the image.
        # 6. Advantage over traditional methods:
        # Unlike methods like Region Proposal Networks (RPN), these proposals are generated without needing to be learned.
        # They provide a dense and uniform coverage of the image at multiple scales.
        backbone_output_memory = self.gen_encoder_output_proposals(
            feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes
        )[0]

        # calculate filtered tokens numbers for each feature map
        # 每个 multi_level_masks 是一个布尔掩码，其中 True 表示无效像素（如填充区域），False 表示有效像素
        reverse_multi_level_masks = [~m for m in multi_level_masks]
        # m.sum((1, 2)) 沿着特征图的高度和宽度（即维度 1 和 2）求和，计算出每张图像中有效的 token（有效像素）的数量
        valid_token_nums = torch.stack(
            [m.sum((1, 2)) for m in reverse_multi_level_masks], -1)
        # 计算出每个特征图中实际需要保留的 token 数量 focus_token_nums.shape = [batch_size, num_feature_levels]
        focus_token_nums = (valid_token_nums * self.level_filter_ratio).int()
        # level_token_nums.shape = (num_feature_levels, )
        level_token_nums = focus_token_nums.max(0)[0]
        # 对每个图像的所有层级的 token 数量进行求和，得到每个图像中需要保留的 token 总数
        # focus_token_nums.shape(batch_size,)
        focus_token_nums = focus_token_nums.sum(-1)

        # from high level to low level
        batch_size = feat_flatten.shape[0]
        selected_score = []
        selected_inds = []
        salience_score = []
        for level_idx in range(spatial_shapes.shape[0] - 1, -1, -1):
            start_index = level_start_index[level_idx]
            end_index = level_start_index[level_idx +
                                          1] if level_idx < spatial_shapes.shape[0] - 1 else None
            # 当前层的特征数据，从 backbone_output_memory 中取出当前层的特征图部分
            level_memory = backbone_output_memory[:, start_index:end_index, :]
            mask = mask_flatten[:, start_index:end_index]
            # update the memory using the higher-level score_prediction
            if level_idx != spatial_shapes.shape[0] - 1:
                # 上采样: score propagates to low level
                upsample_score = torch.nn.functional.interpolate(
                    score,
                    size=spatial_shapes[level_idx].unbind(),
                    mode="bilinear",
                    align_corners=True,
                )
                # upsample_score 的空间维度拉平，将形状从 (batch_size, num_channels, height, width)
                # 改为 (batch_size, num_channels, height × width)，从而更容易与后续的特征进行匹配和处理
                upsample_score = upsample_score.view(
                    batch_size, -1, spatial_shapes[level_idx].prod())
                # transpose(1, 2) 交换维度，将形状从 (batch_size, num_channels, height × width)
                # 变为 (batch_size, height × width, num_channels)
                upsample_score = upsample_score.transpose(1, 2)
                # 这两行代码的目的是将上采样后的显著性得分调整为和当前层特征图的形状相匹配，
                # 使得后续可以通过逐元素相乘的方式将 upsample_score 应用于特征更新
                level_memory = level_memory + level_memory * \
                    upsample_score * self.alpha[level_idx]
            # predict the foreground score of the current layer
            # score.shape = (batch_size, num_tokens_in_level, 1)，这里的 1 表示每个 token 的得分
            # LEARN:学习参数的
            score = self.enc_mask_predictor(level_memory)
            # score.squeeze(-1) 将最后一个维度（即 1）去掉，将 score 的形状变为 (batch_size, num_tokens)
            # For each position where mask is True, the corresponding value in score is replaced with score.min()

            # valid_score.shape = (batch_size, num_tokens_in_level)
            valid_score = score.squeeze(-1).masked_fill(mask, score.min())
            # 将维度 1 和 2 交换，使得 score 的形状变为 (batch_size, 1, num_tokens)
            # score 重新排列为 (batch_size, 1, H, W) 形状，
            # 这样可以与当前层的空间维度对齐。*spatial_shapes[level_idx] 解包了 height 和 width 作为新的维度
            score = score.transpose(1, 2).view(
                batch_size, -1, *spatial_shapes[level_idx])
            # valid_score：经过掩码处理的有效分数，用于计算当前特征图层的 salience 得分
            # score：被调整为符合当前特征图层空间维度的形状，使得后续计算可以正确应用得分

            # get the topk salience index of the current feature map level
            # valid_score 是已经过掩码处理的分数，形状为 (batch_size, num_tokens)
            # level_score：这些 top k token 的得分，形状为 (batch_size, k)
            # level_inds：这些 top k token 的索引，形状同样为 (batch_size, k)
            level_score, level_inds = valid_score.topk(
                level_token_nums[level_idx], dim=1)
            level_inds = level_inds + level_start_index[level_idx]
            salience_score.append(score)
            selected_inds.append(level_inds)
            selected_score.append(level_score)
 
        # 处理完所有的levels，生成salience_score
        # 从低层级到高层级排列, 沿着维度 1（列方向）将所有层级的得分连接起来,
        # (batch_size, total_tokens) 的张量，其中 total_tokens 是所有层级的 token 总数
        selected_score = torch.cat(selected_score[::-1], 1)
        # index 的形状与 selected_score 相同，包含了排序后的索引信息
        index = torch.sort(selected_score, dim=1, descending=True)[1]
        # selected_inds 就包含了按显著性得分从高到低排序的 token 索引，可以用于后续的处理，
        # 如选择最显著的 token 进行进一步分析或特征提取
        # selected_inds.shape = (batch_size, total_selected_tokens)
        selected_inds = torch.cat(selected_inds[::-1], 1).gather(1, index)

        # create layer-wise filtering
        # 实现层级过滤，为不同的网络层选择不同数量的最显著 tokens。
        # 处理显著性分数，将多层级的分数合并成一个统一的前景分数。
        # 对前景分数进行掩码处理，确保填充区域不会影响后续的计算。
        # 这种处理方式允许模型在不同的层中关注不同数量的最显著特征，同时保持了计算的效率和有效性
        num_inds = selected_inds.shape[1]
        # change dtype to avoid shape inference error during exporting ONNX
        cast_dtype = num_inds.dtype if torchvision._is_tracing() else torch.int64
        # 将总 token 数与这些比例相乘，得到每层具体保留的 token 数
        layer_filter_ratio = (
            num_inds * self.layer_filter_ratio).to(cast_dtype)
        # 对每一层，根据计算出的保留数量 r，从 selected_inds 中选择前 r 个索引
        selected_inds = [selected_inds[:, :r] for r in layer_filter_ratio]
        # 反转 salience_score 列表，使其从低层级到高层级排列
        salience_score = salience_score[::-1]
        # 将多层级的显著性分数展平成一个张量
        foreground_score = self.flatten_multi_level(salience_score).squeeze(-1)
        # foreground_score = (batch_size, sum(Hi Wi))
        # sum(Hi Wi) 是所有特征层级的像素总数之和
        foreground_score = foreground_score.masked_fill(
            mask_flatten, foreground_score.min())

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
            feat_unflatten = memory.split(
                spatial_shapes.prod(-1).unbind(), dim=1)
            feat_unflatten = dict((
                i,
                feat.transpose(1, 2).contiguous().reshape(-1,
                                                          self.embed_dim, *spatial_shape),
            ) for i, (feat, spatial_shape) in enumerate(zip(feat_unflatten, spatial_shapes)))
            feat_unflatten = list(self.neck(feat_unflatten).values())
            memory = torch.cat([feat.flatten(2).transpose(1, 2)
                                for feat in feat_unflatten], dim=1)

        # get encoder output, classes and coordinates
        # 关于output_proposals的解释
        # 不同层级之间的宽高是不同的,多尺度特征图的特性，深层特征对应更大的感受野
        # 同一特征层级内，所有位置的初始宽高是相同的
        # 宽高值是相对于整个图像尺寸的
        # 可学习性:
        #    虽然初始值是按层级设置的，但这些值在后续的网络处理中是可以被调整的
        #    Transformer decoder 可以细化这些初始估计
        # output_proposals.shape = (n, sum(hiwi), 4)
        # output_memory.shape = (n, sum(hiwi), c)
        # output_proposals 确实代表了初始的坐标预测，但这些预测通常是基于一些简单的启发式方法或固定的先验生成的
        #                  它们可能不够精确，只是提供了一个粗略的初始估计
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # enc_outputs_class.shape = (n, sum(hiwi), num_classes)
        # encoder_bbox_head
        # 这个模块的作用是预测对初始提议（proposals）的调整或细化。
        # 它不是直接预测最终的坐标，而是预测需要应用到初始提议上的偏移量或修正
        # 这个操作实际上是在执行一个残差学习（residual learning）过程
        # 更容易学习：相比于直接预测绝对坐标，学习相对的调整通常更容易。
        # 保留先验信息：初始提议中包含的先验知识（如可能的目标位置和大小）被保留和利用。
        # 稳定训练：这种残差学习方法通常能提供更稳定的梯度，有助于模型的训练
        # 将最终的坐标值压缩到 [0, 1] 范围内。
        # 这确保了预测的坐标始终在有效的图像范围内
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(
            output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        if torchvision._is_tracing():
            topk = torch.min(torch.tensor(
                self.two_stage_num_proposals * 4), enc_outputs_class.shape[1])
        else:
            topk = min(self.two_stage_num_proposals *
                       4, enc_outputs_class.shape[1])
        # enc_outputs_class.max(-1): -1 参数表示在最后一个维度上操作，也就是类别维度,
        # 这个操作会返回两个张量：最大值和对应的索引
        # enc_outputs_class.max(-1)[0] 的含义是：
        #   对于每个位置，找出所有类别预测分数中的最大值,忽略这个最大值对应的具体类别（索引）
        # topk_scores.shape = (n, topk), topk_index.shape = (n, topk)
        topk_scores, topk_index = torch.topk(
            enc_outputs_class.max(-1)[0], topk, dim=1)
        topk_index = self.nms_on_topk_index(
            topk_scores, topk_index, spatial_shapes, level_start_index, iou_threshold=0.3
        ).unsqueeze(-1)
        # enc_outputs_class.shape = (n, topk, num_classes)
        enc_outputs_class = enc_outputs_class.gather(
            1, topk_index.expand(-1, -1, self.num_classes))
        # enc_outputs_coord.shape = (n, topk, 4)
        enc_outputs_coord = enc_outputs_coord.gather(
            1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()
        # target.shape = (batch_size, num_queries, embed_dim)
        #         处理流程：
        # a. self.tgt_embed.weight 包含了学习到的查询嵌入。
        # b. 这些嵌入被扩展（expand）到批次大小。
        # c. 结果（target）作为解码器的输入查询。
        # 解码器输入的性质：
        # 形状：(batch_size, num_queries, embed_dim)
        # 每个样本有相同的初始查询集。
        # 这些查询在解码过程中会被逐步细化。
        # 5. 为什么使用这种方式：
        # 允许模型学习通用的"目标原型"。
        # 这些原型可以适应不同类型的目标。
        # 提供了一种灵活的方式来指导目标检测过程。
        # 6. 与编码器输出的交互：
        # 解码器将这些查询与编码器的输出（通常是图像特征）结合。
        # 通过注意力机制，查询逐步被更新以定位和分类目标。
        # 这行代码是在创建目标查询（target queries）或对象查询（object queries)
        # multi_level_feats[0].shape[0] 是批次大小（batch size）
        # -1 表示保持原有维度不变
        target = self.tgt_embed.weight.expand(
            multi_level_feats[0].shape[0], -1, -1)

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat(
                [noised_box_query.sigmoid(), reference_points], 1)

        # decoder
        # intput query for content repr: target
        # input query for position: reference_points
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

        # outputs_classes: (batch_size, num_queries, num_classes)
        # outputs_coords: (batch_size, num_queries, 4)
        # enc_outputs_class: (batch_size, num_proposals, num_classes)
        # enc_outputs_coord: (batch_size, num_proposals, 4)
        # salience_score: List of tensors, each with shape (batch_size, 1, Hi, Wi)
        return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, salience_score

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
        topk_scores, topk_index = map(
            lambda x: x.view(-1), (topk_scores, topk_index))

        # get level coordinates for queries and construct boxes for them
        level_index = torch.arange(
            level_start_index.shape[0], device=level_start_index.device)
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
        indices = torchvision.ops.batched_nms(
            coordinates, topk_scores, idxs, iou_threshold)

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


class SalienceTransformerEncoderLayer(nn.Module):
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
        self.pre_attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout, batch_first=True)
        self.pre_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = MultiScaleDeformableAttention(
            embed_dim, n_levels, n_heads, n_points)
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
        src2 = self.linear2(self.dropout2(
            self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    # Here are the shapes of each input:
    # 1. query: (batch_size, num_selected_tokens, embed_dim)
    # This is the selected subset of tokens for the current layer.
    # 2. query_pos: (batch_size, num_selected_tokens, embed_dim)
    # Positional encodings for the selected tokens.
    # 3. value: (batch_size, num_tokens, embed_dim)
    # This is the original full set of tokens, used as values in attention mechanisms.
    # 4. reference_points: (batch_size, num_selected_tokens, num_levels, 2)
    # Reference points for the selected tokens across all feature levels.
    # 5. spatial_shapes: (num_levels, 2)
    # Spatial dimensions of each feature level.
    # 6. level_start_index: (num_levels,)
    # Starting index for each level in the flattened feature representation.
    # 7. query_key_padding_mask: (batch_size, num_tokens)
    # Padding mask for the full set of tokens.
    # 8. score_tgt: (batch_size, num_selected_tokens, num_classes)
    # Class scores for the selected tokens.
    # foreground_pre_layer: (batch_size, num_selected_tokens)
    # Foreground scores for the selected tokens.
    # Key points to note:
    # num_selected_tokens is typically smaller than num_tokens, as it represents the subset of tokens selected based on salience.
    # value and query_key_padding_mask still have the original num_tokens dimension, as they represent the full set of tokens.
    # The other inputs (query, query_pos, reference_points, score_tgt, foreground_pre_layer) have been updated to only include information for the selected tokens.
    # This design allows the layer to focus on processing the most salient tokens while still having access to the full context of the input when needed (through value and query_key_padding_mask).
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
        # score_tgt: (batch_size, num_queries, num_classes)
        # element-wise multiplication between the max class scores and the foreground scores
        # mc_score has shape (batch_size, num_queries)
        # combines class prediction confidence with foreground confidence
        mc_score = score_tgt.max(-1)[0] * foreground_pre_layer
        # select_tgt_index 的形状是 (batch_size, topk_sa)
        select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]
        # 使用 unsqueeze(-1) 将 select_tgt_index 的最后一个维度扩展一个新维度，使其形状变为 (batch_size, topk_sa, 1)
        # expand(-1, -1, self.embed_dim) 将最后一个维度扩展为 embed_dim，从而得到形状为 (batch_size, topk_sa, embed_dim)
        select_tgt_index = select_tgt_index.unsqueeze(
            -1).expand(-1, -1, self.embed_dim)
        select_tgt = torch.gather(query, 1, select_tgt_index)
        select_pos = torch.gather(query_pos, 1, select_tgt_index)
        query_with_pos = key_with_pos = self.with_pos_embed(
            select_tgt, select_pos)
        # 允许模型在特征处理过程中只专注于最重要的目标，并将它们的处理结果整合回原始的特征表示中
        # tgt2.shape = (batch_size, num_selected_tokens, embed_dim)
        # 这种初始化方式有以下几个特点：
        # 位置感知：通过将位置编码加到查询和键中，模型可以考虑空间信息。
        # 特征重用：值张量使用原始特征，保留了完整的特征信息。
        # 查询-键一致性：查询和键使用相同的初始化，这在自注意力机制中很常见。
        # 选择性：只使用最显著的标记，而不是所有的标记。
        tgt2 = self.pre_attention(
            query_with_pos,
            key_with_pos,
            select_tgt,
        )[0]
        # 这个过程的目的是：
        # 信息整合：通过残差连接，将注意力机制的输出与原始输入相结合，保留both原始信息和新处理的信息。
        # 2. 数值稳定：dropout和层归一化有助于训练的稳定性和模型的泛化能力。
        # 3. 选择性更新：只更新被选中的（最显著的）标记，而保持其他标记不变。
        # 4. 维持一致性：虽然只处理了一部分标记，但通过scatter操作，保持了整个query张量的一致性和完整性。
        # 这种方法允许模型在每一层聚焦于最重要的特征，同时保持对整个输入的全面表示。它既提高了计算效率（因为只处理部分标记），
        # 又保持了对整个场景的理解能力。这对于复杂的视觉任务，如目标检测或图像分割，特别有效
        select_tgt = select_tgt + self.pre_dropout(tgt2)
        select_tgt = self.pre_norm(select_tgt)
        # query = query.scatter(1, select_tgt_index, select_tgt)
        # scatter 操作按照 select_tgt_index 指定的索引，将 select_tgt 的值分散到 query 中
        query = query.scatter(1, select_tgt_index, select_tgt)

        # self attention: deformable attention
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

        # 整个过程可以概括为：选择重要目标 -> 预注意力处理 -> 整合回原始查询 ->
        # 自注意力处理 -> FFN处理。这种设计旨在提高模型对显著特征的关注度和处理效率

        return query


class SalienceTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        # learnt background embed for prediction
        self.background_embedding = PositionEmbeddingLearned(
            200, num_pos_feats=self.embed_dim // 2)

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
                torch.linspace(0.5, h - 0.5, h,
                               dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w,
                               dtype=torch.float32, device=device),
                indexing="ij",
            )
            # 这里的归一化是指将特征图的坐标映射到相对的尺度范围内，使得这些坐标在不同大小的特征图之间具有一致性
            ref_y = ref_y.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        # [n, s, l, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


        # memory = self.encoder(
        #     query=feat_flatten,                       (batch_size, num_tokens, embed_dim)
        #     query_pos=lvl_pos_embed_flatten,          (batch_size, num_tokens, embed_dim)
        #     query_key_padding_mask=mask_flatten,      (batch_size, num_tokens)
        #     spatial_shapes=spatial_shapes,            (num_levels, 2)
        #     level_start_index=level_start_index,      (num_levels,)
        #     valid_ratios=valid_ratios,                (batch_size, num_levels, 2)
        #     # salience input
        #     foreground_score=foreground_score,        (batch_size, num_tokens)
        #     focus_token_nums=focus_token_nums,        (batch_size,)
        #     foreground_inds=selected_inds,            List of (batch_size, num_selected_tokens) tensors, one for each layer
        #     multi_level_masks=multi_level_masks,      List of (batch_size, H, W) tensors, one for each level
        # )
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
        # This is particularly useful for operations like deformable attention, 
        # where a point might need to attend to its corresponding locations across different feature scales.
        # reference_points.shape = (batch_size, num_tokens, num_levels, 2)
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        # b、n、s、p 分别表示批次大小、token数量、尺度数量和每个token的参考点坐标
        b, n, s, p = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        value = output = query
        for layer_id, layer in enumerate(self.layers):
            # These lines select the most salient tokens based on foreground_inds for the current layer. 
            # The corresponding query features and positional encodings are gathered.
            # inds_for_query: (batch_size, num_selected_tokens, embed_dim)
            inds_for_query = foreground_inds[layer_id].unsqueeze(
                -1).expand(-1, -1, self.embed_dim)
            # query & query_pos: (batch_size, num_selected_tokens, embed_dim)
            query = torch.gather(output, 1, inds_for_query)
            query_pos = torch.gather(ori_pos, 1, inds_for_query)
            # foreground_pre_layer: (batch_size, num_selected_tokens)
            foreground_pre_layer = torch.gather(
                foreground_score, 1, foreground_inds[layer_id])
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)
            ).view(b, -1, s, p)
            # Score Calculation: (batch_size, num_selected_tokens, num_classes)
            # Shape of score_tgt: (batch_size, num_selected_tokens, num_classes)
            # 根据选出的前景foreground_score query产生每个分类的score
            score_tgt = self.enhance_mcsp(query)

            # Here are the shapes of each input:
            # 1. query: (batch_size, num_selected_tokens, embed_dim)
            # This is the selected subset of tokens for the current layer.
            # 2. query_pos: (batch_size, num_selected_tokens, embed_dim)
            # Positional encodings for the selected tokens.
            # 3. value: (batch_size, num_tokens, embed_dim)
            # This is the original full set of tokens, used as values in attention mechanisms.
            # 4. reference_points: (batch_size, num_selected_tokens, num_levels, 2)
            # Reference points for the selected tokens across all feature levels.
            # 5. spatial_shapes: (num_levels, 2)
            # Spatial dimensions of each feature level.
            # 6. level_start_index: (num_levels,)
            # Starting index for each level in the flattened feature representation.
            # 7. query_key_padding_mask: (batch_size, num_tokens)
            # Padding mask for the full set of tokens.
            # 8. score_tgt: (batch_size, num_selected_tokens, num_classes)
            # Class scores for the selected tokens.
            # foreground_pre_layer: (batch_size, num_selected_tokens)
            # Foreground scores for the selected tokens.
            # Key points to note:
            # num_selected_tokens is typically smaller than num_tokens, as it represents the subset of tokens selected based on salience.
            # value and query_key_padding_mask still have the original num_tokens dimension, as they represent the full set of tokens.
            # The other inputs (query, query_pos, reference_points, score_tgt, foreground_pre_layer) have been updated to only include information for the selected tokens.
            # This design allows the layer to focus on processing the most salient tokens while still having access to the full context of the input when needed (through value and query_key_padding_mask).
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
            # List of batch_size tensors, each with shape (num_tokens, embed_dim)
            outputs = []
            # 这里，我们使用 scatter 操作来更新 output。
            # 重要的是要理解 output 在整个过程中的角色：
            # 它开始时是完整的输入查询。
            # 在每一层，我们从 out put 中选择最显著的标记进行处理。
            # 处理后，我们将更新后的标记放回 output 中的原始位置。
            # 这个更新后的 output 然后被用作下一层的输入。
            # 这种方法允许模型在每一层聚焦于最显著的部分，同时保持对整个输入的完整表示。output 在整个前向传播过程中不断被更新，反映了模型对输入的逐步细化理解。
            # 最后，更新后的 output 作为整个编码器的输出返回。
            for i in range(foreground_inds[layer_id].shape[0]):
                # focus_token_nums[i] ensures we only use the valid (non-padded) indices
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                query_no_pad = query[i][:focus_token_nums[i]]
                outputs.append(
                    output[i].scatter(
                        0,
                        foreground_inds_no_pad.unsqueeze(
                            -1).repeat(1, query.size(-1)),
                        query_no_pad,
                    )
                )
            # Shape: (batch_size, num_tokens, embed_dim)
            output = torch.stack(outputs)

        # add learnt embedding for background
        # NOTE: not break the embeding learning from encoder
        if multi_level_masks is not None:
            background_embedding = [
                self.background_embedding(mask).flatten(2).transpose(1, 2) for mask in multi_level_masks
            ]
            background_embedding = torch.cat(background_embedding, dim=1)
            background_embedding.scatter_(1, inds_for_query, 0)
            background_embedding *= (~query_key_padding_mask).unsqueeze(-1)
            # the output will be foreground (salienece) embedding + background_embedding
            output = output + background_embedding

        return output


class SalienceTransformerDecoderLayer(nn.Module):
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
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True)
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


class SalienceTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(
            2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        self.class_head = nn.ModuleList(
            [nn.Linear(self.embed_dim, num_classes) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList(
            [MLP(self.embed_dim, self.embed_dim, 4, 3) for _ in range(num_layers)])
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
        valid_ratio_scale = torch.cat(
            [valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[
                :, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(
                reference_points_input[:, :, 0, :])
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
            output_coord = self.bbox_head[layer_idx](
                self.norm(query)) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # iterative bounding box refinement
            reference_points = self.bbox_head[layer_idx](
                query) + inverse_sigmoid(reference_points.detach())
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords
