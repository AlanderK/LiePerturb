from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class _LegacyLieAlgebraPerturb(nn.Module):
    """
    自适应李代数扰动模块
    """

    def __init__(self, dim: int, eps=1):
        super().__init__()
        self.eps = eps

        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            QuickGELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        v = self.net(x)  # v(x): [batch, dim]

        # 构造 A(x) = v(x) x^T - x v(x)^T（反对称矩阵）
        Ax = v - (v * x).sum(dim=-1, keepdim=True) * x  # A(x) x

        x_perturbed = x + self.eps * Ax
        return x_perturbed

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LieAlgebraPerturb(nn.Module):
    def __init__(self, dim: int, eps=0.1):
        super().__init__()
        self.eps = eps
        self.unary = nn.Sequential(
            nn.Linear(dim, dim * 4),
            QuickGELU(),
            nn.Linear(dim * 4, dim)
        )
        self.pair = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            QuickGELU(),
            nn.Linear(dim * 4, dim)
        )

    @staticmethod
    def _tangent(vector: torch.Tensor, anchor: torch.Tensor):
        return vector - (vector * anchor).sum(dim=-1, keepdim=True) * anchor

    def forward(self, z_img: torch.Tensor, z_txt: torch.Tensor):
        pair_vector = self.pair(torch.cat([z_img, z_txt], dim=-1))
        delta_img = self._tangent(self.unary(z_txt), z_img) + self._tangent(pair_vector, z_img)
        delta_txt = self._tangent(self.unary(z_img), z_txt) + self._tangent(pair_vector, z_txt)
        return self.eps * delta_img, self.eps * delta_txt


def sphere_exponential_map(z: torch.Tensor, delta: torch.Tensor, eps: float = 1e-6):
    delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(eps)
    direction = delta / delta_norm
    return torch.cos(delta_norm) * z + torch.sin(delta_norm) * direction


class ArtifactAwareFAMBlock(nn.Module):
    def __init__(self, dim: int, image_dim: int, text_dim: int, image_layers: int = 4, text_layers: int = 2):
        super().__init__()
        self.image_mlp = nn.Sequential(
            nn.Linear(image_layers * image_dim, dim * 2),
            QuickGELU(),
            nn.Linear(dim * 2, dim)
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(text_layers * text_dim, dim * 2),
            QuickGELU(),
            nn.Linear(dim * 2, dim)
        )
        self.image_projection = nn.Parameter(torch.empty(dim, dim))
        self.text_projection = nn.Parameter(torch.empty(dim, dim))
        self.lie = LieAlgebraPerturb(dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.image_projection)
        nn.init.eye_(self.text_projection)
        self.image_projection.data.add_(0.02 * torch.randn_like(self.image_projection))
        self.text_projection.data.add_(0.02 * torch.randn_like(self.text_projection))

    def forward(self, image_group: torch.Tensor, text_group: torch.Tensor):
        image_feature = self.image_mlp(image_group)
        text_feature = self.text_mlp(text_group)

        image_projection = self.image_projection.to(dtype=image_feature.dtype, device=image_feature.device)
        text_projection = self.text_projection.to(dtype=text_feature.dtype, device=text_feature.device)
        z_img = F.normalize(image_feature @ image_projection, dim=-1)
        z_txt = F.normalize(text_feature @ text_projection, dim=-1)

        delta_img, delta_txt = self.lie(z_img, z_txt)
        z_img = sphere_exponential_map(z_img, delta_img)
        z_txt = sphere_exponential_map(z_txt, delta_txt)

        image_out = z_img @ image_projection.t()
        text_out = z_txt @ text_projection.t()
        return image_out, text_out

    @staticmethod
    def _orthogonal_loss(projection: torch.Tensor):
        projection = projection.float()
        dim = projection.shape[0]
        identity = torch.eye(projection.shape[1], dtype=projection.dtype, device=projection.device)
        return (projection.t().matmul(projection) - identity).pow(2).sum() / (dim * dim)

    def orthogonal_loss(self):
        return self._orthogonal_loss(self.image_projection) + self._orthogonal_loss(self.text_projection)

    def reorthogonalize_projections(self):
        with torch.no_grad():
            for projection in (self.image_projection, self.text_projection):
                u, _, vh = torch.linalg.svd(projection.data.float(), full_matrices=False)
                projection.copy_((u @ vh).to(dtype=projection.dtype, device=projection.device))


class ArtifactAwareFAM(nn.Module):
    def __init__(
            self,
            dim: int,
            image_dim: int,
            text_dim: int,
            stages: int = 6,
            image_layers_per_stage: int = 4,
            text_layers_per_stage: int = 2
    ):
        super().__init__()
        self.stages = stages
        self.image_layers_per_stage = image_layers_per_stage
        self.text_layers_per_stage = text_layers_per_stage
        self.blocks = nn.ModuleList([
            ArtifactAwareFAMBlock(dim, image_dim, text_dim, image_layers_per_stage, text_layers_per_stage)
            for _ in range(stages)
        ])
        self.image_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.text_residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, image_features, text_features, image_layers, text_layers):
        fam_dtype = self.blocks[0].image_projection.dtype
        fam_device = self.blocks[0].image_projection.device
        image_features = image_features.to(dtype=fam_dtype, device=fam_device)
        text_features = text_features.to(dtype=fam_dtype, device=fam_device)
        image_outputs = []
        text_outputs = []
        for idx, block in enumerate(self.blocks):
            image_start = idx * self.image_layers_per_stage
            text_start = idx * self.text_layers_per_stage
            image_group = torch.cat(
                image_layers[image_start:image_start + self.image_layers_per_stage],
                dim=-1
            ).to(dtype=fam_dtype, device=fam_device)
            text_group = torch.cat(
                text_layers[text_start:text_start + self.text_layers_per_stage],
                dim=-1
            ).to(dtype=fam_dtype, device=fam_device)
            image_out, text_out = block(image_group, text_group)
            image_outputs.append(image_out)
            text_outputs.append(text_out)

        image_delta = torch.stack(image_outputs, dim=0).mean(dim=0)
        text_delta = torch.stack(text_outputs, dim=0).mean(dim=0)
        image_scale = self.image_residual_scale.to(dtype=image_features.dtype, device=image_features.device)
        text_scale = self.text_residual_scale.to(dtype=text_features.dtype, device=text_features.device)
        image_features = image_features + image_scale * image_delta.to(image_features.dtype)
        text_features = text_features + text_scale * text_delta.to(text_features.dtype)
        return image_features, text_features

    def orthogonal_loss(self):
        return torch.stack([block.orthogonal_loss() for block in self.blocks]).mean()

    def reorthogonalize_projections(self):
        for block in self.blocks:
            block.reorthogonalize_projections()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, feature_indices: torch.Tensor = None):
        if feature_indices is not None:
            if feature_indices.ndim != 1 or feature_indices.shape[0] != x.shape[1]:
                raise ValueError(
                    "feature_indices must contain one sequence position per batch item."
                )
            feature_indices = feature_indices.to(device=x.device, dtype=torch.long)
            batch_indices = torch.arange(x.shape[1], device=x.device)

        out = {}
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            if feature_indices is None:
                # The vision transformer uses position 0 as its CLS token.
                layer_feature = x[0]
            else:
                # The text transformer supplies each sample's EOT position so
                # that every intermediate feature represents the full caption.
                layer_feature = x[feature_indices, batch_indices]
            out['layer' + str(idx)] = layer_feature
        return out, x

        # return self.resblocks(x)  # This is the original code 


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))



    def forward_features(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])


        out['before_projection'] = x  

        if self.proj is not None:
            x = x @ self.proj
        out['after_projection'] = x 

        layer_features = [out['layer' + str(idx)] for idx in range(self.transformer.layers)]
        return x, layer_features

    def forward(self, x: torch.Tensor):
        # This only returns CLIP features
        return self.forward_features(x)[0]


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fam = None
        self.fam_supported = (
            not isinstance(vision_layers, (tuple, list))
            and vision_layers == 24
            and image_resolution == 224
            and vision_patch_size == 14
            and transformer_layers == 12
            and embed_dim == 768
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def init_fam(self):
        if not self.fam_supported:
            return
        self.fam = ArtifactAwareFAM(
            dim=self.text_projection.shape[1],
            image_dim=self.visual.transformer.width,
            text_dim=self.transformer.width,
            stages=6,
            image_layers_per_stage=4,
            text_layers_per_stage=2
        )

    def parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if not name.startswith("fam."):
                yield param

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def _encode_image_with_layers(self, image):
        if not hasattr(self.visual, "forward_features"):
            return self.encode_image(image), None
        return self.visual.forward_features(image.type(self.dtype))

    def encode_text(self, text):
        return self._encode_text_with_layers(text)[0]

    def _encode_text_with_layers(self, text):
        # CLIP represents a sentence with its EOT token. EOT has the largest
        # token id in each sequence, so argmax returns one EOT position per item.
        eot_indices = text.argmax(dim=-1)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        layer_outputs, x = self.transformer(x, feature_indices=eot_indices)
        layer_features = [layer_outputs['layer' + str(idx)] for idx in range(self.transformer.layers)]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices] @ self.text_projection

        return x, layer_features

    def encode_image_text(self, image, text):
        with torch.no_grad():
            image_features, image_layers = self._encode_image_with_layers(image)
            text_features, text_layers = self._encode_text_with_layers(text)

        if self.fam is None:
            return image_features, text_features
        if image_layers is None or text_layers is None:
            raise RuntimeError("FAM requires the ViT-L/14 visual transformer and text transformer layer features.")

        return self.fam(
            image_features.detach(),
            text_features.detach(),
            [feature.detach() for feature in image_layers],
            [feature.detach() for feature in text_layers]
        )

    def forward(self, image, text):
        image_features, text_features = self.encode_image_text(image, text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    model.init_fam()
    return model.eval()
