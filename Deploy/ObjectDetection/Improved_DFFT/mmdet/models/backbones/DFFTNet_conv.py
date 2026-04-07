import os

import mmcv

from .DOT_blocks import *
import numpy as np
import cv2
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn

device = torch.device("cuda:0")  # 或者根据你的GPU设备选择合适的设备
# 定义模板图像文件夹路径
# templates_folder = '/home/mayi/data/lmy/ObjectDetection/Improved_DFFT/Template_img/flame_clip/'
templates_folder = '/home/mayi/data/lmy/ObjectDetection/Improved_DFFT/Template_img_kaggle/'
# 定义transforms，包括ToTensor操作
mean = [103.53, 116.28, 123.675]
std = [1.0, 1.0, 1.0]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


def load_templates(templates_folder):
    template_images = []
    # 获取模板图像文件路径列表
    template_files = glob.glob(os.path.join(templates_folder, '*.jpg'))

    # 遍历模板图像文件列表
    for template_file in template_files:
        # 加载模板图像
        template_bgr = cv2.imread(template_file)
        if template_bgr.ndim == 2:
            # 增加通道维度
            template_bgr = np.expand_dims(template_bgr, axis=2)
            # 重复通道值，使其成为一个完整的通道
            template_bgr = np.repeat(template_bgr, 3, axis=2)
        template_bgr = template_bgr.astype(np.float32) #(height, width, channels)
        # 假设 image 是一个形状为 (height, width, channels) 的多维数组
        image_transposed = np.transpose(template_bgr, (2, 0, 1))
        # template = mmcv.imnormalize(template_bgr, mean, std, False)
        template_tensor = torch.tensor(image_transposed).to(device)
        normalize = transforms.Normalize(mean=mean, std=std)
        normalized_template_tensor = normalize(template_tensor)
        gray_tensor = torch.mean(normalized_template_tensor, dim=0, keepdim=True)  # 对三个通道取均值，保持维度 [1, H, W]
        template_images.append(gray_tensor[0])

        # template_tensor = transform(template_bgr).to(device)  # 形状为 (3, H, W)
        # gray_tensor = torch.mean(template_tensor, dim=0, keepdim=True)  # 对三个通道取均值，保持维度 [1, H, W]
        # template_images.append(gray_tensor[0])

    return template_images

def template_res(img, template):
    # print(normalized_img.shape, normalized_template.shape)
    # print("normalized_img-------", torch.max(normalized_img),torch.min(normalized_img))
    # print("normalized_template-------", torch.max(normalized_template), torch.min(normalized_template))

    mean_img = torch.mean(img)
    mean_template = template.mean()

    # 计算源图像和模板图像的标准差
    std_img = img.std()
    std_template = template.std()

    # 归一化源图像和模板图像
    normalized_img = (img - mean_img) / std_img
    normalized_template = (template - mean_template) / std_template
    # 计算归一化的互相关系数
    correlation = F.conv2d(normalized_img, normalized_template, padding=0, stride=1)

    # 找到最大的归一化互相关系数值
    max_correlation = torch.max(correlation)
    # print("correlation-------", max_correlation, torch.min(correlation))

    # 归一化互相关系数图像
    normalized_correlation = correlation / max_correlation
    # 计算源图像和模板图像的均值
    mean_res = torch.mean(normalized_correlation)
    # 计算源图像和模板图像的标准差
    std_res = normalized_correlation.std()

    # 归一化源图像和模板图像
    normalized_res = (normalized_correlation - mean_res) / std_res

    # # 打印最大的归一化互相关系数值
    # print("Max normalized cross-correlation:", max_correlation.item())

    return normalized_res  # 返回归一化的互相关系数矩阵

def conv_match(template_images, img_tensors, th, tw): # 新模板
    processed_images = []
    # _, _, img_h, img_w = img_tensors.shape
    # 加载目标图像
    for template in template_images: #Tensor[h,w]
        h, w = template.shape
        # 如果模板尺寸大于目标图像尺寸，则缩小为当前尺寸的一半
        if h > th or w > tw:
            template = cv2.resize(template.cpu().numpy(), (tw // 2, th // 2))
            # h, w = img_h // 2, img_w // 2
            template = torch.from_numpy(template).to(device)
        match = template_res(img_tensors, template.unsqueeze(0).unsqueeze(0))
        #---> tensor(batch,1,n,m)--> tensor(batch,n,m)
        resized_match = F.interpolate(match, size=(th, tw), mode='bilinear', align_corners=False).squeeze(dim=1)
        processed_images.append(resized_match)      # list(temp_num)--every item:tensor(batch,th,tw)
    # tensor(temp_num,batch,th,tw)
    fusion_match = torch.stack(processed_images, dim=1).to(device) # tensor(batch,temp_num,th,tw)
    # tensor(temp_num,batch,th,tw)-->(batch,th,tw)
    # fusion_match = fusion_match.mean(dim=1)# tensor(batch,th,tw)
    # # fusion_match, _ = torch.max(fusion_match, dim=0)  # tensor(batch,th,tw)
    # # 扩展通道维度
    # fusion_match = fusion_match.unsqueeze(1)  # 在第1维插入通道维度 tensor(batch,1,th,tw)

    # 重复通道数据
    # fusion_match = fusion_match.repeat(1, 3, 1, 1)  # 在第2维和第3维重复3次

    return fusion_match #tensor(batch,temp_num,th,tw)





def b16(n, activation, resolution=224):
    #  Conv2d_BN(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000)
    return torch.nn.Sequential(
        Conv2d_BN(3, n, 3, 2, 1, resolution=resolution),
        activation(),)

@BACKBONES.register_module()
class DFFTNet_conv(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 alldepths=[3, 3, 19, 3],
                 num_heads=[4, 4, 7, 12],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 2, 4, 6),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 crossca_position=[1, 2, 3],
                 ):
        super().__init__()

        print("depths:", depths)
        print("num_heads", num_heads)
        print("crossca_position:", crossca_position)

        self.template_images = load_templates(templates_folder)

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = num_heads[0] * 32
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages


        self.b16 = b16(self.embed_dim, torch.nn.Hardswish)


        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=self.embed_dim, embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # if self.ape:
        #     pretrain_img_size = to_2tuple(pretrain_img_size)
        #     patch_size = to_2tuple(patch_size)
        #     patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
        #
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)
        #
        #     self.absolute_pos_embed_temp = nn.Parameter(
        #         torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
        #     trunc_normal_(self.absolute_pos_embed_temp, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop_temp = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        layer_dim = int(num_heads[0] * 32)


        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(num_heads[i_layer]*32)
            layer_dimout = int(num_heads[i_layer+1]*32) if (i_layer < self.num_layers - 1) else int(num_heads[i_layer]*32)
            layer = DOTBlock(
                dim=layer_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                alldepths=alldepths[i_layer])
            self.layers.append(layer)
            if i_layer in crossca_position:
                saablock = SAAModule(layer_dim, 2, torch.nn.Hardswish, resolution=224, drop_path=0.,)
                self.layers.append(saablock)
            if (i_layer < self.num_layers - 1):
                downsample = PatchMerging(dim=layer_dim, dimout = layer_dimout, norm_layer=norm_layer)
                self.layers.append(downsample)

        num_features = [int(num_heads[i]*32) for i in range(self.num_layers)]
        self.num_features = num_features
        self.upsample_2 = torch.nn.Upsample(scale_factor=2)

        # 使用1x1的卷积层和BatchNorm层对通道数进行压缩和调整
        # self.temp_conv1x1 = nn.Conv2d(1, num_features[0], kernel_size=1)
        # self.temp_conv_2d = nn.Conv2d(len(self.template_images), len(self.template_images), kernel_size=1, bias=False)
        # self.temp_bn_2d = nn.BatchNorm2d(len(self.template_images))

        self.temp_conv_bn_2d = Conv2d_BN(len(self.template_images), len(self.template_images), ks=1, resolution=224)
        self.temp_conv1x1 = nn.Conv1d(num_features[0]+len(self.template_images), num_features[0], kernel_size=1, bias=False)
        self.temp_bn = nn.BatchNorm1d(num_features[0])

        self.links = nn.ModuleList()
        for i_layer in range(1, self.num_layers):
            if i_layer == 1:
                layer_dim = 128
            else:
                layer_dim = self.num_features[-1]
            saeblock = SAEBlock(layer_dim, 2, torch.nn.Hardswish, resolution=224, drop_path=0.)
            self.links.append(saeblock)

        self.out_norm = norm_layer(self.num_features[-1])

        saaconv = []
        saaconv.append(nn.Sequential(nn.Conv2d(self.num_features[0], self.num_features[1], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[1]), nn.ReLU6(inplace=True)))
        saaconv.append(nn.Sequential(nn.Conv2d(self.num_features[1], self.num_features[1], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[1]), nn.ReLU6(inplace=True)))
        saaconv.append(nn.Sequential(nn.Conv2d(self.num_features[1], self.num_features[2], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[2]), nn.ReLU6(inplace=True)))
        saaconv.append(nn.Sequential(nn.Conv2d(self.num_features[2], self.num_features[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[3]), nn.ReLU6(inplace=True)))
        for idx in range(4):
            layer_name = f'saaconv{idx}'
            self.add_module(layer_name, saaconv[idx])   


        saeconv = []
        saeconv.append(nn.Sequential(nn.Conv2d(self.num_features[1], 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU6(inplace=True)))
        saeconv.append(nn.Sequential(nn.Conv2d(self.num_features[2], 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU6(inplace=True)))
        saeconv.append(nn.Sequential(nn.Conv2d(128, self.num_features[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[3]), nn.ReLU6(inplace=True)))
        saeconv.append(nn.Sequential(nn.Conv2d(self.num_features[3], self.num_features[3], 3, 1, 1, bias=False), nn.BatchNorm2d(self.num_features[3]), nn.ReLU6(inplace=True)))
        for idx in range(4):
            layer_name = f'saeconv{idx}'
            self.add_module(layer_name, saeconv[idx])   

        self._freeze_stages()



    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()

            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv1d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Conv2d):
            #     trunc_normal_(m.weight, std=0.02)
            #     # nn.init.kaiming_normal_(m.weight, a=0, nonlinearity='relu')
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1.0)
            #     nn.init.constant_(m.bias, 0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, filename_list):
        # template操作
        mean_rgb = torch.mean(x, dim=1, keepdim=True)

        # print("x------------")
        # print(x.shape)
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        x = self.b16(x) #(batch, 3, 800, 1216)→(batch, 3, )
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        temp = conv_match(self.template_images, mean_rgb, Wh, Ww)#tensor(batch,temp_num,th,tw)
        # print("temp.shape", temp.shape, Wh, Ww)
        # if temp.shape[2]==len(self.template_images):
        #     temp = temp.permute(0, 2, 1, 3)  # 将第二维和第三维互换位置
        #     temp = F.interpolate(temp, size=(Wh, Ww), mode='bilinear', align_corners=False)
        # temp = self.temp_conv_2d(temp)
        # temp = self.temp_bn_2d(temp)
        temp = self.temp_conv_bn_2d(temp)
        # print("temp.shape", temp.shape, Wh, Ww) #torch.Size([6, temp_num, 128, 168]) 128 168

        # if self.ape:
        #     absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
        #     x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        # else:
        #     x = x.flatten(2).transpose(1, 2)

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        dot_feature, dot_HW = [], []
        saa_feature = []
        channel = [self.num_features[1], self.num_features[1], self.num_features[2], self.num_features[3]]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DOTBlock):
                x, H, W = layer(x, Wh, Ww)
                B, _, C = x.shape
                cross_x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                conv_layer = getattr(self, f'saaconv{len(dot_feature)}')
                if len(dot_feature) < 2:
                    cross_x = conv_layer(cross_x)
                dot_feature.append(cross_x.contiguous().view(B, channel[len(dot_feature)], -1).transpose(-2, -1))
                dot_HW.append([H, W])
            elif isinstance(layer, SAAModule):
                if i == len(self.layers)-1:
                    last_layer = True
                    x, _ = layer(dot_feature[-2:], dot_HW[-2:], last_layer=last_layer)
                    saa_feature.append(x)
                else:
                    link_x, x = layer(dot_feature[-2:], dot_HW[-2:])
                    saa_feature.append(link_x)
                    if len(dot_feature) > 1:
                        cross_x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                        conv_layer = getattr(self, f'saaconv{len(dot_feature)}')
                        cross_x = conv_layer(cross_x)
                        dot_feature[-1] =cross_x.contiguous().view(B, channel[len(dot_feature)], -1).transpose(-2, -1)
                    else:
                        dot_feature[-1] = link_x
            elif isinstance(layer, PatchMerging):
                x = layer(x, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
        
        saa_feature.append(dot_feature[-1])


        # template操作
        # temp_Wh, temp_Ww = temp.size(2), temp.size(3)  # temp与x尺寸相同
        # if self.ape:
        #     temp_absolute_pos_embed = F.interpolate(self.absolute_pos_embed_temp, size=(temp_Wh, temp_Ww), mode='bicubic')
        #     temp = (temp + temp_absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        # else:
        #     temp = temp.flatten(2).transpose(1, 2)

        temp = temp.flatten(2).transpose(1, 2)
        temp = self.pos_drop_temp(temp)

        # addlink2:
        dot_feature = saa_feature
        # print("前dot_feature[0].shape", dot_feature[0].shape, dot_feature[0])

        # print("temp.shape, dot_feature[0].shape", temp.shape, dot_feature[0].shape)
        # torch.Size([6, 13600, 3])
        # torch.Size([6, 13600, 128])
        # 将temp和dot_stage1在通道维度上拼接得到c，形状为torch.Size([batch, 6720, 256])
        temp_dot = torch.cat((temp, dot_feature[0]), dim=2)
        # print("temp_dot.shape", temp_dot.shape)

        # 将c传入卷积层和BatchNorm层，形状变为torch.Size([batch, 6720, 128])
        temp_dot = temp_dot.transpose(1, 2)  # 调整维度顺序以适应卷积层的输入要求[batch, 256, 6720]
        temp_dot = self.temp_conv1x1(temp_dot) # [batch, 128, 6720]
        temp_dot = self.temp_bn(temp_dot)
        temp_dot = temp_dot.transpose(1, 2) # [batch, 6720, 128]

        dot_feature[0] = temp_dot
        # print("后dot_feature[0].shape", dot_feature[0].shape, dot_feature[0])

        channel = [128, 128, self.num_features[3], self.num_features[3]]
        for i in range(2):
            H, W = dot_HW[i]  # 获取第i个DOTBlock输出特征的高度和宽度
            B, _, C = dot_feature[i].shape  # 获取第i个DOTBlock输出特征的形状，其中B为batch size，C为通道数
            cross_x = dot_feature[i].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            # 将第i个DOTBlock输出特征进行形状变换，将通道维度放到第二个位置，高度和宽度放到第三和第四个位置
            # 结果保存在cross_x中
            conv_layer = getattr(self, f'saeconv{i}')  # 获取名为'saeconv{i}'的卷积层
            cross_x = conv_layer(cross_x)  # 使用'saeconv{i}'对cross_x进行卷积操作
            dot_feature[i] = cross_x.contiguous().view(B, channel[i], -1).transpose(-2, -1)
            # 将处理后的cross_x再次进行形状变换，将通道维度恢复为channel[i]，并将高度和宽度展平，保存在dot_feature[i]中

        for i in range(len(self.links)):
            H, W = dot_HW[i + 1]  # 获取第i+1个DOTBlock输出特征的高度和宽度
            B, _, C = dot_feature[i + 1].shape  # 获取第i+1个DOTBlock输出特征的形状，其中B为batch size，C为通道数
            layer = self.links[i]  # 获取第i个SAEBlock
            if i == len(self.links) - 1:
                last_layer = True
                x, _ = layer(dot_feature[i:i + 2], dot_HW[i:i + 2], last_layer=last_layer)
                # 如果是最后一个SAEBlock，则将第i和i+1个DOTBlock的特征以及对应的高度和宽度传递给SAEBlock进行处理
            else:
                conv_layer = getattr(self, f'saeconv{i + 2}')  # 获取名为'saeconv{i+2}'的卷积层
                last_layer = False
                _, x = layer(dot_feature[i:i + 2], dot_HW[i:i + 2], last_layer=last_layer)
                # 如果不是最后一个SAEBlock，则将第i和i+1个DOTBlock的特征以及对应的高度和宽度传递给SAEBlock进行处理
                x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                # 将处理后的特征进行形状变换，将通道维度放到第二个位置，高度和宽度放到第三和第四个位置
                x = conv_layer(x)  # 使用'saeconv{i+2}'对x进行卷积操作
                dot_feature[i + 1] = x.contiguous().view(B, channel[i + 2], -1).transpose(-2, -1)
                # 将处理后的x再次进行形状变换，将通道维度恢复为channel[i+2]，并将高度和宽度展平，保存在dot_feature[i+1]中

        x = self.out_norm(x)
        x = x.view(-1, dot_HW[2][0], dot_HW[2][1], self.num_features[-1]).permute(0, 3, 1, 2).contiguous()
        return tuple([x])

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DFFTNet_conv, self).train(mode)
        self._freeze_stages()
# 
