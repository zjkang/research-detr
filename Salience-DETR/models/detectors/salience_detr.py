from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.denoising import GenerateCDNQueries
from models.bricks.losses import sigmoid_focal_loss
from models.detectors.base_detector import DNDETRDetector


class SalienceCriterion(nn.Module):
    def __init__(
        self,
        limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
        noise_scale: float = 0.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.limit_range = limit_range
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, foreground_mask, targets, feature_strides, image_sizes):
        # foreground_mask：模型生成的前景掩码，通常是一个形状为 [batch_size, 1, H, W] 的张量列表，每个元素对应不同特征层级的掩码
        # 可学习部分：前景掩码是通过模型生成的，这意味着它由模型的可学习参数（如卷积层、注意力机制等）计算而来。
        # 在训练过程中，模型参数通过反向传播学习，使得生成的 foreground_mask 更加准确地反映前景对象
        # mask_targets：这是基于目标边界框生成的掩码，用于计算与模型生成的 foreground_mask 之间的差异。
        # 它是计算损失的参考标准，并不是可学习的参数
        gt_boxes_list = []
        for t, (img_h, img_w) in zip(targets, image_sizes):
            # targets: 是归一化坐标
            # 根据图像的实际宽度和高度，将边界框坐标从归一化坐标缩放到实际像素坐标
            boxes = t["boxes"]
            # 原始格式通常为 (cx, cy, w, h)，即中心点坐标和宽高
            boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            # [img_w, img_h, img_w, img_h] 对应 (x_min, y_min, x_max, y_max)
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            gt_boxes_list.append(boxes * scale_factor)

        # 主要功能是根据不同特征层级（feature levels）的特征图，生成与目标检测任务中真实边界框（ground truth boxes）
        # 对应的显著性掩码（salience masks）。这些掩码用于指导模型学习在特定位置检测对象
        mask_targets = []
        for level_idx, (mask, feature_stride) in enumerate(zip(foreground_mask, feature_strides)):
            feature_shape = mask.shape[-2:]
            coord_x, coord_y = self.get_pixel_coordinate(feature_shape, feature_stride, device=mask.device)
            masks_per_level = []
            for gt_boxes in gt_boxes_list:
                # 掩码表示特征图上的哪些位置与目标边界框匹配（通常取值为0到1之间的浮点数，表示显著性概率）
                mask = self.get_mask_single_level(coord_x, coord_y, gt_boxes, level_idx)
                masks_per_level.append(mask)
            # torch.stack 将多个掩码张量沿着一个新维度堆叠起来，形成形状为 [num_gt_boxes, H, W] 的张量
            masks_per_level = torch.stack(masks_per_level)
            mask_targets.append(masks_per_level)
        # 拼接所有特征层级的掩码目标
        # torch.cat(mask_targets, dim=1) 的作用是沿着维度 1 将 mask_targets 中的所有张量进行拼接
        mask_targets = torch.cat(mask_targets, dim=1)
        # 将所有层级的前景掩码展平并拼接在一起
        foreground_mask = torch.cat([e.flatten(-2) for e in foreground_mask], -1)
        foreground_mask = foreground_mask.squeeze(1)
        num_pos = torch.sum(mask_targets > 0.5 * self.noise_scale).clamp_(min=1)

        # Focal Loss 计算前景显著性损失，考虑了目标的不平衡性，并根据像素坐标与目标边界的距离来调整损失
        salience_loss = (
            sigmoid_focal_loss(
                foreground_mask,
                mask_targets,
                num_pos,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * foreground_mask.shape[1]
        )
        return {"loss_salience": salience_loss}

    def get_pixel_coordinate(self, feature_shape, stride, device):
        height, width = feature_shape
        coord_y, coord_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
            indexing="ij",
        )
        coord_y = coord_y.reshape(-1)
        coord_x = coord_x.reshape(-1)
        return coord_x, coord_y

    def get_mask_single_level(self, coord_x, coord_y, gt_boxes, level_idx):
        # gt_label: (m,) gt_boxes: (m, 4)
        # coord_x: (h*w, )
        left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]  # (h*w, m)
        top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1]
        right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None]
        bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None]
        # 四个方向的距离合并为一个张量，形状为 (h*w, m, 4)
        border_distances = torch.stack(
            [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
            dim=-1,
        )  # [h*w, m, 4]

        # the foreground queries must satisfy two requirements:
        # 1. the quereis located in bounding boxes
        # 2. the distance from queries to the box center match the feature map stride
        min_border_distances = torch.min(border_distances, dim=-1)[0]  # [h*w, m]
        max_border_distances = torch.max(border_distances, dim=-1)[0]
        # min_border_distances 表示每个像素到每个边界框最小的边界距离，如果最小距离大于零，说明该像素在边界框内，
        # 生成 mask_in_gt_boxes 布尔掩码，表示像素是否位于边界框内
        mask_in_gt_boxes = min_border_distances > 0
        min_limit, max_limit = self.limit_range[level_idx]
        # mask_in_level 用来筛选那些距离在当前特征层有效范围内的像
        mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit)
        # mask_pos 通过 mask_in_gt_boxes 和 mask_in_level 两个掩码相与，
        # 筛选出同时满足位于边界框内且符合层级距离要求的像素
        mask_pos = mask_in_gt_boxes & mask_in_level

        # scale-independent salience confidence
        # delta_x 和 delta_y 计算的是像素相对于边界框中心的归一化偏移值，confidence 代表像素的几何置信度，取值范围为 [0, 1]，
        # 越接近边界框中心的像素置信度越
        row_factor = left_border_distance + right_border_distance
        col_factor = top_border_distance + bottom_border_distance
        delta_x = (left_border_distance - right_border_distance) / row_factor
        delta_y = (top_border_distance - bottom_border_distance) / col_factor
        confidence = torch.sqrt(delta_x**2 + delta_y**2) / 2

        # confidence_per_box 的维度为 [h*w, m]
        confidence_per_box = 1 - confidence
        confidence_per_box[~mask_in_gt_boxes] = 0

        # process positive coordinates
        if confidence_per_box.numel() != 0:
            mask = confidence_per_box.max(-1)[0]
        else:
            mask = torch.zeros(coord_y.shape, device=confidence.device, dtype=confidence.dtype)

        # process negative coordinates
        # 对于不满足 mask_pos 条件的像素，将它们的掩码值设为 0
        mask_pos = mask_pos.long().sum(dim=-1) >= 1
        mask[~mask_pos] = 0

        # add noise to add randomness
        mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)
        return mask


# SalienceDETR has the architecture similar to FocusDETR
class SalienceDETR(DNDETRDetector):
    def __init__(
        # model structure
        self,
        # 负责提取图像的多层次特征
        backbone: nn.Module,
        # 进一步处理从 backbone 提取的特征
        neck: nn.Module,
        position_embedding: nn.Module,
        transformer: nn.Module,
        # 用于计算损失的标准，如分类和定位损失
        criterion: nn.Module,
        # 用于推理阶段，将模型输出转换为实际检测框
        postprocessor: nn.Module,
        # 一个自定义损失，用于额外的焦点损失计算
        # focus detr, salience detr 感觉这个可以用类似的方式
        focus_criterion: nn.Module,
        # model parameters
        num_classes: int = 91,
        num_queries: int = 900,
        denoising_nums: int = 100,
        # model variants
        aux_loss: bool = True,
        min_size: int = None,
        max_size: int = None,
    ):
        super().__init__(min_size, max_size)
        # define model parameters
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        embed_dim = transformer.embed_dim

        # define model structures
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.denoising_generator = GenerateCDNQueries(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=embed_dim,
            denoising_nums=denoising_nums,
            label_noise_prob=0.5,
            box_noise_scale=1.0,
        )
        self.focus_criterion = focus_criterion

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # images: 输入图像张量列表, images come with batch (b,h,w)
        # 目标检测的真实标签（包含边界框和类别信息）
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        # 标记实际图像内容的区域为 0，即 mask[img_id, :image_size[0], :image_size[1]] = 0，
        # 表示该区域是有效的图像部分，而填充的区域仍然保持为 1，表示无效的填充区域
        images, targets, mask = self.preprocess(images, targets)

        # extract features
        multi_level_feats = self.backbone(images.tensors)
        multi_level_feats = self.neck(multi_level_feats)

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feature in multi_level_feats:
            # mask[None]：先在维度0增加一个新的维度，这样 mask 就变成了 1 x H x W 的形式，以便与插值操作匹配
            multi_level_masks.append(F.interpolate(mask[None], size=feature.shape[-2:]).to(torch.bool)[0])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        if self.training:
            # collect ground truth for denoising generation
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
            noised_results = self.denoising_generator(gt_labels_list, gt_boxes_list)
            noised_label_query = noised_results[0]
            noised_box_query = noised_results[1]
            attn_mask = noised_results[2]
            denoising_groups = noised_results[3]
            max_gt_num_per_image = noised_results[4]
        else:
            noised_label_query = None
            noised_box_query = None
            attn_mask = None
            denoising_groups = None
            max_gt_num_per_image = None

        # feed into transformer
        # 将多尺度特征、掩码和位置嵌入输入到 transformer，输出类别和边界框预测结果，以及用于焦点损失的前景掩码
        outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            noised_label_query,
            noised_box_query,
            attn_mask=attn_mask,
        )
        # hack implementation for distributed training
        outputs_class[0] += self.denoising_generator.label_encoder.weight[0, 0] * 0.0

        # denoising postprocessing
        if denoising_groups is not None and max_gt_num_per_image is not None:
            dn_metas = {
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
            }
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

            # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # compute loss
            loss_dict = self.criterion(output, targets)
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            # compute focus loss
            feature_stride = [(
                images.tensors.shape[-2] / feature.shape[-2],
                images.tensors.shape[-1] / feature.shape[-1],
            ) for feature in multi_level_feats]
            focus_loss = self.focus_criterion(foreground_mask, targets, feature_stride, images.image_sizes)
            loss_dict.update(focus_loss)

            # loss reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = dict((k, loss_dict[k] * weight_dict[k]) for k in loss_dict.keys() if k in weight_dict)
            return loss_dict

        detections = self.postprocessor(output, original_image_sizes)
        return detections
