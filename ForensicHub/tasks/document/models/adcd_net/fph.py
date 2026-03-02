"""
FPH (Feature Pyramid Handler) module for ADCD-Net
Handles DCT coefficient encoding and multi-scale feature extraction
"""

import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

# EfficientNet-style configurations
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'
])

GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'
])

global_params = GlobalParams(
    width_coefficient=1.8, depth_coefficient=2.6, image_size=528,
    dropout_rate=0.0, num_classes=1000, batch_norm_momentum=0.99,
    batch_norm_epsilon=0.001, drop_connect_rate=0.0, depth_divisor=8,
    min_depth=None, include_top=True
)


def get_width_and_height_from_size(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, (list, tuple)):
        return x
    raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


def drop_connect(inputs, p, training):
    """Drop connect implementation."""
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MemoryEfficientSwish(nn.Module):
    """Memory efficient Swish activation."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions with static same padding."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                 pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_conv2d(image_size=None):
    """Get Conv2d with same padding based on image size."""
    return lambda in_channels, out_channels, kernel_size, **kwargs: Conv2dStaticSamePadding(
        in_channels, out_channels, kernel_size, image_size=image_size, **kwargs
    )


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block."""

    def __init__(self, block_args, global_params, image_size=25):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio

        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, bias=False
        )
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class AddCoord(nn.Module):
    """Add coordinate channels to input tensor."""

    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_c, yy_c = torch.meshgrid(
            torch.arange(x_dim, dtype=input_tensor.dtype),
            torch.arange(y_dim, dtype=input_tensor.dtype),
            indexing='ij'
        )
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        xx_c = xx_c.expand(batch_size, 1, x_dim, y_dim)
        yy_c = yy_c.expand(batch_size, 1, x_dim, y_dim)
        ret = torch.cat((input_tensor, xx_c, yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class FPH(nn.Module):
    """
    Feature Pyramid Handler for ADCD-Net.
    Processes DCT coefficients and quantization tables to extract multi-scale features
    for document tampering detection.

    Returns:
        as_feat: Alignment score feature (B, 256, H/8, W/8)
        ms_dct_feats: Multi-scale DCT features list
    """

    def __init__(self, dct_feat_dim=256):
        super(FPH, self).__init__()
        self.dct_feat_dim = dct_feat_dim

        # DCT coefficient embedding (0-20 range after clipping)
        self.dct_embed = nn.Embedding(21, 21)
        self.dct_embed.weight.data = torch.eye(21)  # One-hot encoding

        # Quantization table embedding
        self.qt_embed = nn.Embedding(64, 16)

        # Initial DCT processing
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=21, out_channels=64, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.addcoords = AddCoord(with_r=True)

        # Main feature extraction with MBConv blocks
        repeats = (1, 1, 1)
        in_channels = (256, 256, 256)

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=35, out_channels=256, kernel_size=8, stride=8, padding=0, bias=False),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
            MBConvBlock(BlockArgs(
                num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6,
                input_filters=in_channels[0], output_filters=in_channels[1], se_ratio=0.25, id_skip=True
            ), global_params),
            MBConvBlock(BlockArgs(
                num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6,
                input_filters=in_channels[1], output_filters=in_channels[1], se_ratio=0.25, id_skip=True
            ), global_params),
            MBConvBlock(BlockArgs(
                num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6,
                input_filters=in_channels[1], output_filters=in_channels[1], se_ratio=0.25, id_skip=True
            ), global_params),
        )

        # Multi-scale feature extractors
        self.ms_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 48, 3, 1, 1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 48, 3, 1, 1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ),
        ])

    def forward(self, dct, qtable):
        """
        Args:
            dct: DCT coefficients tensor (B, H, W) with values 0-20
            qtable: Quantization table (B, 1, 8, 8) or (B, 8, 8)

        Returns:
            as_feat: Feature for alignment score prediction (B, 256, H/8, W/8)
            ms_dct_feats: List of multi-scale DCT features
        """
        # Ensure qtable has correct shape
        if len(qtable.shape) == 3:
            qtable = qtable.unsqueeze(1)

        # Embed DCT coefficients
        x = self.dct_embed(dct.long())  # (B, H, W, 21)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 21, H, W)
        x = self.conv2(self.conv1(x))  # (B, 16, H, W)

        B, C, H, W = x.shape

        # Reshape for QT modulation
        x_reshaped = x.reshape(B, C, H // 8, 8, W // 8, 8)
        x_reshaped = x_reshaped.permute(0, 1, 3, 5, 2, 4)  # (B, C, 8, 8, H/8, W/8)

        # Embed quantization table
        qt_embedded = self.qt_embed(qtable.squeeze(1).long())  # (B, 8, 8, 16)
        qt_embedded = qt_embedded.permute(0, 3, 1, 2).unsqueeze(-1).unsqueeze(-1)  # (B, 16, 8, 8, 1, 1)

        # Modulate with QT
        x_modulated = x_reshaped * qt_embedded  # (B, 16, 8, 8, H/8, W/8)
        x_modulated = x_modulated.permute(0, 1, 4, 2, 5, 3)  # (B, 16, H/8, 8, W/8, 8)
        x_modulated = x_modulated.reshape(B, C, H, W)

        # Concatenate with original and add coordinates
        x_cat = torch.cat([x_modulated, x], dim=1)  # (B, 32, H, W)
        x_cat = self.addcoords(x_cat)  # (B, 35, H, W)

        # Main feature extraction
        as_feat = self.conv0(x_cat)  # (B, 256, H/8, W/8)

        # Generate multi-scale features
        ms_dct_feats = []
        for i, conv in enumerate(self.ms_convs):
            scale = 2 ** i
            if scale > 1:
                feat = F.interpolate(as_feat, scale_factor=1/scale, mode='bilinear', align_corners=False)
            else:
                feat = as_feat
            ms_dct_feats.append(conv(feat))

        return as_feat, ms_dct_feats


if __name__ == "__main__":
    fph = FPH()
    dct = torch.randint(0, 21, (2, 256, 256))
    qt = torch.randint(0, 64, (2, 1, 8, 8))
    as_feat, ms_feats = fph(dct, qt)
    print(f"Alignment score feature shape: {as_feat.shape}")
    for i, f in enumerate(ms_feats):
        print(f"Multi-scale feature {i} shape: {f.shape}")