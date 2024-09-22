
import torch
import torch.nn as nn

import numpy as np
PI = np.pi

class Hough(nn.Module):

    def __init__(self, angle=90, R2_list=[4, 64, 256, 1024],
                 num_classes=80, region_num=9, vote_field_size=17,
                 voting_map_size_w=128, voting_map_size_h=128, model_v1=False):
        super(Hough, self).__init__()
        self.angle = angle # how many angles each 360 (angle)
        self.R2_list = R2_list # region distanct to the center (layer)
        self.region_num = region_num # total # of regions
        self.num_classes = num_classes # total # of classes
        self.vote_field_size = vote_field_size
        self.deconv_filter_padding = int(self.vote_field_size / 2)
        self.voting_map_size_w = voting_map_size_w
        self.voting_map_size_h = voting_map_size_h
        self.model_v1 = model_v1
        self.deconv_filters = self._prepare_deconv_filters()


    def  _prepare_deconv_filters(self):
        # 那么，为什么改变 voting_map_size 会影响结果？
        # 主要原因在于输入数据的变化和整个处理流程：
        # a. 输入数据的变化：
        # 更大的 voting_map_size 意味着输入到 Hough 模块的特征图更大。
        # 这会影响 calculate_logmap 函数的输出，进而影响后续的权重计算。
        # b. 权重计算的变化：
        # 更大的 voting_map_size 会改变 logmap_onehot 的大小和内容。
        # 这导致归一化过程产生不同的 weights
        # 虽然提取的区域大小（由 vote_field_size 决定）不变，但提取的内容会因为 weights 的变化而不同

        half_w = int(self.voting_map_size_w / 2)
        half_h = int(self.voting_map_size_h / 2)

        vote_center = torch.tensor([half_h, half_w]).cuda()

        # vote_field (H,W,R) with one-hot: (128, 128, 9)
        logmap_onehot = self.calculate_logmap((self.voting_map_size_h, self.voting_map_size_w), vote_center)
        # normalize on (H,W) -> (H,W,R): (128, 128, 9)
        # 每个位置的 9 维向量现在被归一化，使得在整个图像上，每个区域的权重总和为 1
        weights = logmap_onehot / \
                        torch.clamp(torch.sum(torch.sum(logmap_onehot, dim=0), dim=0).float(), min=1.0)

        start_x = half_h - int(self.vote_field_size/2)
        stop_x  = half_h + int(self.vote_field_size/2) + 1

        start_y = half_w - int(self.vote_field_size/2)
        stop_y  = half_w + int(self.vote_field_size/2) + 1

        '''This if-block only applies for my two pretrained models. Please ignore this for your own trainings.'''
        if self.model_v1 and self.region_num==17 and self.vote_field_size==65:
            start_x -=1
            stop_x -=1
            start_y -=1
            stop_y -=1

        # (vote_field_size,vote_field_size,R)
        # -> (R,vote_field_size,vote_field_size)
        # -> (region_num, 1, vote_field_size, vote_field_size): (9, 1, 17, 17)
        # 每个区域（总共 region_num 个）都有一个独立的滤波器
        deconv_filters = weights[start_x:stop_x, start_y:stop_y,:].permute(2,0,1).view(self.region_num, 1,
            self.vote_field_size, self.vote_field_size)
        # W shape: (self.num_classes * region_num, 1, vote_field_size, vote_field_size)
        # (9 * self.num_classes, 1, 17, 17) = (720, 1, 17, 17)
        W = nn.Parameter(deconv_filters.repeat(self.num_classes, 1, 1, 1))
        W.requires_grad = False

        layers = []
        # 分组卷积（Grouped Convolution）是一种卷积操作，其中卷积核和输入通道被分组，允许独立的卷积在不同的通道组上执行。
        # 分组卷积可以减少计算复杂度，并且在深度学习中被用于构建更高效的模型，例如在 ResNeXt 和 MobileNet 中广泛使用
        # 每一个组就是每一个类 独立卷积计算
        # 得到每一个类的特征图
        deconv_kernel = nn.ConvTranspose2d(
            in_channels=self.region_num*self.num_classes,
            out_channels=1*self.num_classes,
            kernel_size=self.vote_field_size,
            padding=self.deconv_filter_padding,
            groups=self.num_classes,
            bias=False)

        with torch.no_grad():
            # deconv_kernel.weight shape the same is W
            # 结果格式符合 PyTorch 卷积层的权重格式：(out_channels, in_channels, height, width)
            deconv_kernel.weight = W

        layers.append(deconv_kernel)

        # output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
        # output_size = (128 - 1) * 1 - 2 * 8 + 17 + 0
        #     = 127 - 16 + 17
        #     = 128
        return nn.Sequential(*layers)

    def generate_grid(self, h, w):
        x = torch.arange(0, w).float().cuda()
        y = torch.arange(0, h).float().cuda()
        grid = torch.stack([x.repeat(h), y.repeat(w, 1).t().contiguous().view(-1)], 1)
        return grid.repeat(1, 1).view(-1, 2)

    # zk: generate log-polar vote field - HxWxR (regions)
    def calculate_logmap(self, im_size, center, angle=90, R2_list=[4, 64, 256, 1024]):
        points = self.generate_grid(im_size[0], im_size[1])  # [x,y]
        total_angles = 360 / angle

        # check inside which circle
        y_dif = points[:, 1].cuda() - center[0].float()
        x_dif = points[:, 0].cuda() - center[1].float()

        xdif_2 = x_dif * x_dif
        ydif_2 = y_dif * y_dif
        sum_of_squares = xdif_2 + ydif_2

        # find angle
        arc_angle = (torch.atan2(y_dif, x_dif) * 180 / PI).long()

        arc_angle[arc_angle < 0] += 360

        angle_id = (arc_angle / angle).long() + 1

        c_region = torch.ones(xdif_2.shape, dtype=torch.long).cuda() * len(R2_list)

        for i in range(len(R2_list) - 1, -1, -1):
            region = R2_list[i]
            c_region[(sum_of_squares) <= region] = i

        results = angle_id + (c_region - 1) * total_angles
        results[results < 0] = 0

        # shape: (H,W), results[H][W] = id
        results.view(im_size[0], im_size[1])

        logmap = results.view(im_size[0], im_size[1])
        logmap_onehot = torch.nn.functional.one_hot(logmap.long(), num_classes=17).float()
        logmap_onehot = logmap_onehot[:, :, :self.region_num]

        # (H, W, self.region_num), 预定义的区域数量
        # 对于图像中的每个像素 (i, j)，logmap_onehot[i, j, :] 是一个 one-hot 向量
        #  one-hot 向量表示该像素属于哪个对数极坐标区域
        return logmap_onehot

    # voting_map:(h,w,r,c)
    def forward(self, voting_map, targets=None):

        if self.model_v1:
            batch_size, channels, width, height = voting_map.shape
            voting_map = voting_map.view(batch_size, self.region_num, self.num_classes, width, height)
            voting_map = voting_map.permute(0, 2, 1, 3, 4)
            voting_map = voting_map.reshape(batch_size, -1, width, height)

         # voting_map: h*w*r*c
        # heatmap: (h,w,c)
        heatmap = self.deconv_filters(voting_map)

        return heatmap

