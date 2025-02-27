from __future__ import annotations
import torch.nn as nn
import torch

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F
import einops


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(out_channels // 2, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(out_channels // 2, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            DoubleConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up2d(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2d(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat((x2, x1), dim=1)
        return self.conv(x)


class OutConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class UNet2d(nn.Module):
    def __init__(self, in_cha=3, out_cha=3, features=[32, 32, 32], bilinear=True, first=False, final=False):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.first = first
        self.final = final

        factor = 2 if bilinear else 1

        if self.first:
            self.inc = (DoubleConv2d(in_cha, features[0]))
        in_channels = features[0]
        for feature in features[1:]:
            self.downs.append(Down2d(in_channels, feature))
            in_channels = feature

        for feature in reversed(features[:-1]):
            self.ups.append(Up2d(in_channels, feature // factor))
            in_channels = feature

        if self.final:
            self.outc = (OutConv2d(features[0], out_cha))

    def forward(self, x):
        skip_connections = []
        if self.first:
            x = self.inc(x)

        for down in self.downs:
            skip_connections.append(x)
            x = down(x)

        for up, skip in zip(self.ups, reversed(skip_connections)):
            x = up(x, skip)

        if self.final:
            x = self.outc(x)

        return x


class uC_skip(nn.Module):
    def __init__(self, in_channels, out_channels, channels, bilinear, first, final):  # feat==channel
        super().__init__()
        self.unet = UNet2d(in_channels, out_channels, channels, bilinear, first, final)

    def forward(self, x):
        sp = x.shape[-1]
        x1 = einops.rearrange(x, 'B C D H W -> (B W) C D H')
        x1 = self.unet(x1)
        x1 = einops.rearrange(x1, '(B W) C D H -> B C D H W', W=sp)
        return x1


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class TripletMixerMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        """
        Args:
            dim: Model dimension.
            d_state: State-space size for the SSM.
            d_conv: Convolution width.
            expand: Expansion factor for the intermediate dimension.
        """
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # SSM and convolutions
        self.mamba = Mamba(
            d_model=dim,  # State-space model
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.conv1d_1 = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            groups=dim,
            padding="same",
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            groups=dim,
            padding="same",
        )
        self.conv1d_3 = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            groups=dim,
            padding="same"
        )

        # Output projections
        self.linear_out = nn.Linear(dim, dim)
        self.linear_skip = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, channels, spatial_dims...).

        Returns:
            Output tensor of the same shape as the input.
        """
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim

        # Flatten spatial dimensions
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        # Normalization
        x_norm = self.norm(x_flat)

        # SSM path
        x_ssm = self.mamba(x_norm)

        # Conv1D paths
        x_conv1 = F.silu(self.conv1d_1(x_norm.transpose(-1, -2))).transpose(-1, -2)
        x_conv2 = F.silu(self.conv1d_2(x_norm.transpose(-1, -2))).transpose(-1, -2)
        x_conv3 = F.silu(self.conv1d_3(x_norm.transpose(-1, -2))).transpose(-1, -2)
        # Combine all paths
        x_combined = x_ssm + x_conv1 + x_conv2 + x_conv3

        # Final projections
        out = self.linear_out(x_combined) + self.linear_skip(x_norm)

        # Reshape back to original dimensions
        out = out.transpose(-1, -2).reshape(B, C, *img_dims)

        # Residual connection
        out = out + x_skip
        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[TripletMixerMambaLayer(dim=dims[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class TriXMamba(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                )

        self.encoder1 = uC_skip(in_channels=self.in_chans, out_channels=self.feat_size[0],
                                channels=[self.feat_size[0],
                                          self.feat_size[1],
                                          self.feat_size[2],
                                          self.feat_size[3]], bilinear=False, first=True, final=True)
        self.encoder2 = uC_skip(in_channels=self.feat_size[0], out_channels=self.feat_size[1],
                                channels=[self.feat_size[0],
                                          self.feat_size[1],
                                          self.feat_size[2],
                                          self.feat_size[3]], bilinear=False, first=False, final=True)

        self.encoder3 = uC_skip(in_channels=self.feat_size[1], out_channels=self.feat_size[2],
                                channels=[self.feat_size[1],
                                          self.feat_size[2],
                                          self.feat_size[3]], bilinear=False, first=False, final=True)
        self.encoder4 = uC_skip(in_channels=self.feat_size[2], out_channels=self.feat_size[3],
                                channels=[self.feat_size[2],
                                          self.feat_size[3]], bilinear=False, first=False, final=True)

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        return self.out(out)
