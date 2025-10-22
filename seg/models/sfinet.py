import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import timm

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, 
                 norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, 
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, 
                 norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, 
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, 
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class MSCFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.dwconv1 = nn.Conv2d(in_features, in_features//2, 1, 1)
        self.dwconv2 = nn.Conv2d(in_features, in_features//4, kernel_size=3, stride=1, padding=1)
        self.dwconv3 = nn.Conv2d(in_features, in_features//4, kernel_size=7, stride=1, padding=3)

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x3 = self.dwconv3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SCARF(nn.Module):
    """
    Spatial-Channel Attentive Residual Fusion (SCARF) module.
    
    This module efficiently fuses frequency-enhanced CNN features and self-attention features
    through parallel, structurally complementary attention paths.
    
    Args:
        in_channels (int): Number of input channels for each feature map
        out_channels (int): Number of output channels
        reduction_ratio (int): Reduction ratio for the channel attention MLP (default: 4)
    """
    
    def __init__(self, in_channels, out_channels=None, reduction_ratio=4):
        super(SCARF, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.initial_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        
        # Channel Path components
        self.channel_gmp = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * reduction_ratio),  # Expand layer
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * reduction_ratio, in_channels),   # Reduce layer
            nn.Sigmoid()
        )
        
        # Spatial Path components
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.spatial_activation = nn.Sigmoid()
        
        # Fusion components
        self.fusion_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.fusion_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize module weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_c, x_a):
        """
        Forward pass of SCARF module.
        
        Args:
            x_c (torch.Tensor): Frequency-enhanced CNN features [B, C, H, W]
            x_a (torch.Tensor): Frequency-enhanced Self-attention features [B, C, H, W]
            
        Returns:
            torch.Tensor: Fused features [B, C, H, W]
        """
        B, C, H, W = x_c.shape
        
        # Step 1: Concatenate and linear projection
        x_concat = torch.cat([x_c, x_a], dim=1)  # [B, 2C, H, W]
        x = self.initial_conv(x_concat)  # [B, C, H, W]
        
        # Step 2: Channel Path
        # Global Max Pooling
        z = self.channel_gmp(x)  # [B, C, 1, 1]
        z_flat = z.view(B, C)  # [B, C]
        
        # MLP with bottleneck structure
        s_c = self.channel_mlp(z_flat)  # [B, C]
        s_c = s_c.view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # Apply channel attention
        x_ch = x * s_c  # [B, C, H, W]
        
        # Step 3: Spatial Path
        # 3x3 convolution and sigmoid activation
        s_s = self.spatial_conv(x)  # [B, C, H, W]
        s_s = self.spatial_activation(s_s)  # [B, C, H, W]
        
        # Apply spatial attention
        x_sp = x * s_s  # [B, C, H, W]
        
        # Step 4: Fusion and Residual Connection
        # Element-wise addition of re-calibrated features
        x_fused = x_ch + x_sp  # [B, C, H, W]
        
        # Refinement with 1x1 convolution
        x_refined = self.fusion_conv1(x_fused)  # [B, C, H, W]
        
        # Residual connection
        x_residual = x_refined + x  # [B, C, H, W]
        
        # Final output
        y = self.fusion_conv2(x_residual)  # [B, out_channels, H, W]
        
        return y

class DBSFIAttention(nn.Module):
    """
    Dual Branch Spatial Frequency Interaction Attention
    """
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size
        self.dim = dim

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=3)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))
        self.wx = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.wy = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.relative_pos_embedding = relative_pos_embedding

        # for compute frequency weight
        self.frequency_weight_sa = nn.Parameter(torch.tensor(0.5))
        # for compute local convolution weight
        self.frequency_weight_cnn = nn.Parameter(torch.tensor(0.5))
        
        self.HWFGM = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_heads))

        self.conv_freq_attn_cnn = nn.Conv2d(self.num_heads, dim, kernel_size=1, bias=False)

        self.fc1 = nn.Linear(dim, dim // 4)
        self.fc2 = nn.Linear(dim // 4, dim)
        self.sigmoid = nn.Sigmoid()

        self.fusion = SCARF(in_channels=dim, out_channels=dim)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)


    def extract_dct_features_torch(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # batch, token数, channel数
        device = x.device

        x_mean = x.mean(dim=-1)  # [B, N]
        feat_size = int(torch.ceil(torch.sqrt(torch.tensor(N, dtype=torch.float32))))
        pad_len = feat_size ** 2 - N
        if pad_len > 0:
            padded = F.pad(x_mean, (0, pad_len), "constant", 0.0)
        else:
            padded = x_mean
        padded = padded.view(B, feat_size, feat_size)

        def dct_2d_torch(img):
            img_ext = torch.cat([img, img.flip(dims=[-1])], dim=-1)
            img_ext = torch.cat([img_ext, img_ext.flip(dims=[-2])], dim=-2)
            fft_res = torch.fft.fft2(img_ext)
            dct_res = fft_res.real[:, :feat_size, :feat_size]
            return dct_res

        dct_coeffs = dct_2d_torch(padded)
        dct_flat = dct_coeffs.reshape(B, -1)[:, :N]
        dct_features = torch.clamp(dct_flat, -10.0, 10.0)
        dct_norm = torch.norm(dct_features, p=2, dim=1, keepdim=True) + 1e-5
        dct_features = dct_features / dct_norm

        return dct_features

    def compute_frequency_weights_cnn(self, x):
        B, C, H, W = x.shape
        f_cnn_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)  # [B, C]
        
        x_bnc = x.view(B, C, H*W).permute(0, 2, 1).contiguous()  # [B, N, C]
        dct_features = self.extract_dct_features_torch(x_bnc)  # [B_, N ]
        f_dct_norm = dct_features / (dct_features.norm(dim=1, keepdim=True) + 1e-6)  # [B, H*W]
        f_dct_embed = F.adaptive_avg_pool1d(f_dct_norm.unsqueeze(1), self.dim).squeeze(1)  # [B, C]
        
        context = f_cnn_pool * f_dct_embed  # element-wise cross modulation, [B, C]

        att = self.fc1(context)
        att = F.relu(att, inplace=True)
        att = self.fc2(att)
        att = self.sigmoid(att).view(B, C, 1, 1)  # [B, C, 1, 1]

        out = x * att
        return out

    def compute_frequency_weights_sa(self, x):
        """
        Compute attention weights based on frequency domain features
        """
        try:
            dct_features = self.extract_dct_features_torch(x)  # [B_, N]

            x_avg = x.mean(dim=1)  # [B_, C]
            head_weights = self.HWFGMs(x_avg)  # [B_, num_heads]

            N = x.size(1)
            dct_features = dct_features.unsqueeze(1)  # [B_, 1, N]
            dct_features = dct_features.repeat(1, self.num_heads, 1)  # [B_, num_heads, N]

            head_weights = head_weights.unsqueeze(-1)  # [B_, num_heads, 1]
            weighted_dct = dct_features * head_weights  # [B_, num_heads, N]

            freq_attn = torch.bmm(weighted_dct.transpose(1, 2), weighted_dct)  # [B_, N, N]

            sum_attn = freq_attn.sum(dim=-1, keepdim=True)
            sum_attn = torch.max(sum_attn, torch.ones_like(sum_attn) * 1e-5)
            freq_attn = freq_attn / sum_attn

            freq_attn = torch.clamp(freq_attn, 0.0, 1.0)

            return freq_attn.unsqueeze(1)

        except Exception as e:
            B_, N = x.size(0), x.size(1)
            return torch.ones(B_, 1, N, N, device=x.device)
            
    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape

        # get frequency attention and feature map
        freq_attn_cnn = self.compute_frequency_weights_cnn(x)  # [B_, 1, N, N], [B, C, H, W]

        # local convolution attention branch
        local_spatial_attn = self.local1(x)
        out_cnn = self.local2(x)
        adaptive_cnn_weight = torch.sigmoid(self.frequency_weight_cnn)
        cnn_attn = (1 - adaptive_cnn_weight) * local_spatial_attn + adaptive_cnn_weight * freq_attn_cnn
        out_cnn = cnn_attn * out_cnn

        # sa attention branch
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
        # print(q.shape, k.shape, v.shape)
        bb, hh, nn, cc = v.shape
        freq_attn_sa = self.compute_frequency_weights_sa(v.view(bb*hh, nn, cc)).view(bb, hh, nn, nn)
        adaptive_sa_weight = torch.sigmoid(self.frequency_weight_sa)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn += relative_position_bias.unsqueeze(0)

        combined_attn = (1 - adaptive_sa_weight) * attn + adaptive_sa_weight * freq_attn_sa
        attn = combined_attn.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        wx, wy = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')), \
                  self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out_sa = self.wx * wx + self.wy * wy
        # print(out_cnn.shape, out_sa.shape)
        out = self.fusion(out_cnn, out_sa)


        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out

class DBSFIFormerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DBSFIAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MSCFFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, 
                          act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, decode_channels//16, kernel_size=1),
            nn.ReLU6(),
            Conv(decode_channels//16, decode_channels, kernel_size=1),
            nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = DBSFIFormerBlock(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = DBSFIFormerBlock(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = DBSFIFormerBlock(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class SFINet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=2
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        # print(encoder_channels)  # [64, 128, 256, 512]
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)  # torch.Size([1, 64, 128, 128]) torch.Size([1, 128, 64, 64]) torch.Size([1, 256, 32, 32]) torch.Size([1, 512, 16, 16])
        x = self.decoder(res1, res2, res3, res4, h, w)
        return x