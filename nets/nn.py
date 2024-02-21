import copy
import math
from os import makedirs
from os.path import exists

import numpy
import torch
from torch.nn import functional
from torchvision import transforms

from utils import util


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=1E-6)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv(ch, ch, torch.nn.GELU('tanh'), k=3, s=1, p=1)
        self.conv2 = Conv(ch, ch, torch.nn.Identity(), k=3, s=1, p=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class MBConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, s, r, fused=True):
        super().__init__()

        if fused:
            features = [Conv(in_ch, r * in_ch, torch.nn.GELU('tanh'), k=3, s=s, p=1),
                        Conv(r * in_ch, out_ch, torch.nn.Identity())]
        else:
            features = [torch.nn.Conv2d(in_ch, r * in_ch, kernel_size=1),
                        torch.nn.GELU('tanh'),
                        torch.nn.Conv2d(r * in_ch, r * in_ch, kernel_size=3, stride=s, padding=1, groups=r * in_ch),
                        torch.nn.GELU('tanh'),
                        Conv(r * in_ch, out_ch, torch.nn.Identity())]
        self.add_m = s == 1 and in_ch == out_ch
        self.res_m = torch.nn.Sequential(*features)

    def forward(self, x):
        y = self.res_m(x)
        return x + y if self.add_m else y


class LiteMLA(torch.nn.Module):
    """
    Lightweight Multiscale Linear Attention
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=None,
                 heads_ratio=1.0,
                 dim=8,
                 scales=(5,),
                 eps=1.0e-15):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        self.dim = dim
        self.qkv = torch.nn.Conv2d(in_channels, 3 * total_dim, kernel_size=1, bias=False)
        self.aggreg = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(3 * total_dim, 3 * total_dim,
                                                                               kernel_size=scale,
                                                                               padding=scale // 2,
                                                                               groups=3 * total_dim, bias=False),
                                                               torch.nn.Conv2d(3 * total_dim, 3 * total_dim,
                                                                               kernel_size=1,
                                                                               groups=3 * heads, bias=False))
                                           for scale in scales])
        self.kernel_func = torch.nn.ReLU(inplace=False)

        self.proj = Conv(total_dim * (1 + len(scales)), out_channels, torch.nn.Identity())

    @torch.cuda.amp.autocast(enabled=False)
    def relu_linear_att(self, qkv):
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W,), )
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (qkv[..., 0: self.dim], qkv[..., self.dim: 2 * self.dim], qkv[..., 2 * self.dim:],)

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        v = functional.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x):
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        return x + self.proj(self.relu_linear_att(multi_scale_qkv))


class ViTBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 heads_ratio=1.0,
                 dim=32,
                 expand_ratio=6,
                 scales=(5,)):
        super().__init__()
        self.context_module = LiteMLA(in_channels=in_channels,
                                      out_channels=in_channels,
                                      heads_ratio=heads_ratio,
                                      dim=dim,
                                      scales=scales)
        self.local_module = MBConv(in_channels, in_channels, s=1, r=expand_ratio, fused=False)

    def forward(self, x):
        x = self.context_module(x)
        return self.local_module(x)


class EfficientViT(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.GELU('tanh'), k=3, s=2, p=1))
        for i in range(depth[0]):
            self.p1.append(Residual(width[1]))
        # p2/4
        self.p2.append(MBConv(width[1], width[2], s=2, r=16))
        for i in range(depth[1]):
            self.p2.append(MBConv(width[2], width[2], s=1, r=4))
        # p3/8
        self.p3.append(MBConv(width[2], width[3], s=2, r=16))
        for i in range(depth[2]):
            self.p3.append(MBConv(width[3], width[3], s=1, r=4))
        # p4/16
        self.p4.append(MBConv(width[3], width[4], s=2, r=16, fused=False))
        for i in range(depth[3]):
            self.p4.append(MBConv(width[4], width[4], s=1, r=4, fused=False))
        # p5/32
        self.p5.append(MBConv(width[4], width[5], s=2, r=24, fused=False))
        for i in range(depth[4]):
            self.p5.append(ViTBlock(width[5]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p5, p4, p3


class Neck(torch.nn.Module):
    def __init__(self, in_channels, n):
        super().__init__()
        self.p5 = torch.nn.Sequential(Conv(in_channels[0], 256, torch.nn.Identity()),
                                      torch.nn.Upsample(size=(64, 64), mode="bicubic", align_corners=False))
        self.p4 = torch.nn.Sequential(Conv(in_channels[1], 256, torch.nn.Identity()),
                                      torch.nn.Upsample(size=(64, 64), mode="bicubic", align_corners=False))
        self.p3 = torch.nn.Sequential(Conv(in_channels[2], 256, torch.nn.Identity()),
                                      torch.nn.Upsample(size=(64, 64), mode="bicubic", align_corners=False))

        modules = []
        for _ in range(n):
            modules.append(MBConv(256, 256, s=1, r=1))

        self.output = torch.nn.Sequential(*modules,
                                          torch.nn.Conv2d(256, 256, kernel_size=1))
        self.norm_weight = torch.nn.parameter.Parameter(torch.empty(256, device=None, dtype=None))
        self.norm_bias = torch.nn.parameter.Parameter(torch.empty(256, device=None, dtype=None))

    def forward(self, x):
        p5, p4, p3 = x
        p5 = self.p5(p5)
        p4 = self.p4(p4)
        p3 = self.p3(p3)
        x = self.output(p3 + p4 + p5)

        y = x - torch.mean(x, dim=1, keepdim=True)
        y = y / torch.sqrt(torch.square(y).mean(dim=1, keepdim=True) + 1E-6)
        return y * self.norm_weight.view(1, -1, 1, 1) + self.norm_bias.view(1, -1, 1, 1)


class LayerNorm(torch.nn.Module):
    def __init__(self, num_channels, eps=1E-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PositionEmbedding(torch.nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)), )

    def _pe_encoding(self, coords):
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * numpy.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input, image_size):
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class PromptEncoder(torch.nn.Module):
    def __init__(self,
                 embed_dim,
                 image_embedding_size,
                 input_image_size,
                 mask_in_chans,
                 activation=torch.nn.GELU):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbedding(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [torch.nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = torch.nn.ModuleList(point_embeddings)
        self.not_a_point_embed = torch.nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = torch.nn.Sequential(torch.nn.Conv2d(1,
                                                                    mask_in_chans // 4,
                                                                    kernel_size=2, stride=2),
                                                    LayerNorm(mask_in_chans // 4),
                                                    activation(),
                                                    torch.nn.Conv2d(mask_in_chans // 4,
                                                                    mask_in_chans,
                                                                    kernel_size=2, stride=2),
                                                    LayerNorm(mask_in_chans),
                                                    activation(),
                                                    torch.nn.Conv2d(mask_in_chans,
                                                                    embed_dim, kernel_size=1))
        self.no_mask_embed = torch.nn.Embedding(1, embed_dim)

    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points, labels, padding):
        points = points + 0.5  # Shift to center of pixel
        if padding:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes):
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks):
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    @staticmethod
    def _get_batch_size(points, boxes, masks):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self):
        return self.point_embeddings[0].weight.device

    def forward(self, points, boxes, masks):
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, padding=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1])

        return sparse_embeddings, dense_embeddings


class MLP(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 sigmoid_output=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = functional.sigmoid(x)
        return x


class MaskDecoder(torch.nn.Module):
    def __init__(self, *,
                 transformer_dim,
                 transformer,
                 num_multi_mask_outputs=3,
                 activation=torch.nn.GELU,
                 iou_head_depth=3,
                 iou_head_hidden_dim=256):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multi_mask_outputs = num_multi_mask_outputs

        self.iou_token = torch.nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multi_mask_outputs + 1
        self.mask_tokens = torch.nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = torch.nn.Sequential(torch.nn.ConvTranspose2d(transformer_dim,
                                                                             transformer_dim // 4,
                                                                             kernel_size=2, stride=2),
                                                    LayerNorm(transformer_dim // 4),
                                                    activation(),
                                                    torch.nn.ConvTranspose2d(transformer_dim // 4,
                                                                             transformer_dim // 8,
                                                                             kernel_size=2, stride=2),
                                                    activation())
        self.output_hypernetworks_mlps = torch.nn.ModuleList([MLP(transformer_dim,
                                                                  transformer_dim,
                                                                  transformer_dim // 8, 3)
                                                              for _ in range(self.num_mask_tokens)])

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(self,
                image_embeddings,
                image_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                multimask_output):
        masks, iou_pred = self.predict_masks(image_embeddings=image_embeddings,
                                             image_pe=image_pe,
                                             sparse_prompt_embeddings=sparse_prompt_embeddings,
                                             dense_prompt_embeddings=dense_prompt_embeddings)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(self,
                      image_embeddings,
                      image_pe,
                      sparse_prompt_embeddings,
                      dense_prompt_embeddings):
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


class Attention(torch.nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_heads,
                 downsample_rate=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = torch.nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = torch.nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = torch.nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = torch.nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class TwoWayAttentionBlock(torch.nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_heads,
                 mlp_dim=2048,
                 activation=torch.nn.ReLU,
                 attention_downsample_rate=2,
                 skip_first_layer_pe=False):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = torch.nn.LayerNorm(embedding_dim, eps=1E-6)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads,
                                                   downsample_rate=attention_downsample_rate)
        self.norm2 = torch.nn.LayerNorm(embedding_dim, eps=1E-6)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(embedding_dim, mlp_dim),
                                       activation(),
                                       torch.nn.Linear(mlp_dim, embedding_dim))
        self.norm3 = torch.nn.LayerNorm(embedding_dim, eps=1E-6)

        self.norm4 = torch.nn.LayerNorm(embedding_dim, eps=1E-6)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads,
                                                   downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(torch.nn.Module):
    def __init__(self,
                 depth,
                 embedding_dim,
                 num_heads,
                 mlp_dim,
                 activation=torch.nn.ReLU,
                 attention_downsample_rate=2):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = torch.nn.ModuleList()

        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(embedding_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    mlp_dim=mlp_dim,
                                                    activation=activation,
                                                    attention_downsample_rate=attention_downsample_rate,
                                                    skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads,
                                                   downsample_rate=attention_downsample_rate)
        self.norm_final_attn = torch.nn.LayerNorm(embedding_dim, eps=1E-6)

    def forward(self,
                image_embedding,
                image_pe,
                point_embedding):
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layer-norm
        for layer in self.layers:
            queries, keys = layer(queries=queries,
                                  keys=keys,
                                  query_pe=point_embedding,
                                  key_pe=image_pe, )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class EfficientViTSAM(torch.nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, backbone, neck, prompt_encoder, mask_decoder, image_size=(1024, 512)):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.image_size = image_size

        self.transform = transforms.Compose([util.SamResize(self.image_size[1]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                                                                  std=[58.395 / 255, 57.12 / 255, 57.375 / 255], ),
                                             util.SamPad(self.image_size[1])])

        self.features = None
        self.input_size = None
        self.is_image_set = False
        self.original_size = None

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x

    def postprocess_masks(self,
                          masks,
                          input_size,
                          original_size):
        masks = functional.interpolate(masks,
                                       size=(self.image_size[0], self.image_size[0]),
                                       mode="bilinear", align_corners=False)
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = functional.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    @property
    def device(self):
        return self.parameters().__next__().device

    def reset(self):
        self.features = None
        self.input_size = None
        self.is_image_set = False
        self.original_size = None

    def apply_coords(self, coords):
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        coords = copy.deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes):
        boxes = boxes.reshape(-1, 2, 2)
        boxes = self.apply_coords(boxes)
        return boxes.reshape(-1, 4)

    @torch.inference_mode()
    def set_image(self, image, image_format="RGB"):
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.image_format:
            image = image[..., ::-1]

        self.reset()

        self.original_size = image.shape[:2]
        self.input_size = util.ResizeLongestSide.get_preprocess_shape(*self.original_size,
                                                                      long_side_length=self.image_size[0])

        x = self.transform(image).unsqueeze(dim=0).to(self.device)
        self.features = self.forward(x)
        self.is_image_set = True

    def predict(self,
                point_coords=None,
                point_labels=None,
                box=None,
                mask_input=None,
                multi_mask_output=True):
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (numpy.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (numpy.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (numpy.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (numpy.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where for SAM, H=W=256.
          multi_mask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multi_mask_output=False can give better results.

        Returns:
          (numpy.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (numpy.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (numpy.ndarray): An array of shape CxHxW, where C is the number of masks and H=W=256.
           These low resolution logits can be passed to a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = self.apply_coords(point_coords)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.apply_boxes(box)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_fn(coords_torch,
                                                                labels_torch,
                                                                box_torch,
                                                                mask_input_torch,
                                                                multi_mask_output)

        masks = masks[0].detach().cpu().numpy()
        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return masks, iou_predictions, low_res_masks

    @torch.inference_mode()
    def predict_fn(self,
                   point_coords=None,
                   point_labels=None,
                   boxes=None,
                   mask_input=None,
                   multi_mask_output=True):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points,
                                                                  boxes=boxes,
                                                                  masks=mask_input)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(image_embeddings=self.features,
                                                           image_pe=self.prompt_encoder.get_dense_pe(),
                                                           sparse_prompt_embeddings=sparse_embeddings,
                                                           dense_prompt_embeddings=dense_embeddings,
                                                           multimask_output=multi_mask_output, )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        return masks > self.mask_threshold, iou_predictions, low_res_masks


def build_sam_l0(image_size=512):
    if not exists('./weights'):
        makedirs('./weights')
    if not exists('./weights/sam_l0.pt'):
        url = 'https://github.com/jahongir7174/EfficientViTSAM/releases/download/v0.0.1/sam_l0.pt'
        torch.hub.download_url_to_file(url=url, dst='./weights/sam_l0.pt')
    backbone = EfficientViT(width=[3, 32, 64, 128, 256, 512], depth=[1, 1, 1, 4, 4])
    neck = Neck(in_channels=[512, 256, 128], n=4)
    model = EfficientViTSAM(backbone=backbone,
                            neck=neck,
                            prompt_encoder=PromptEncoder(embed_dim=256,
                                                         image_embedding_size=(64, 64),
                                                         input_image_size=(1024, 1024),
                                                         mask_in_chans=16),
                            mask_decoder=MaskDecoder(num_multi_mask_outputs=3,
                                                     transformer=TwoWayTransformer(depth=2,
                                                                                   embedding_dim=256,
                                                                                   mlp_dim=2048,
                                                                                   num_heads=8),
                                                     transformer_dim=256,
                                                     iou_head_depth=3,
                                                     iou_head_hidden_dim=256),
                            image_size=(1024, image_size))

    state_dict = torch.load('./weights/sam_l0.pt')['state_dict']
    model.load_state_dict(state_dict)
    return model


def build_sam_l1(image_size=512):
    if not exists('./weights'):
        makedirs('./weights')
    if not exists('./weights/sam_l1.pt'):
        url = 'https://github.com/jahongir7174/EfficientViTSAM/releases/download/v0.0.1/sam_l1.pt'
        torch.hub.download_url_to_file(url=url, dst='./weights/sam_l1.pt')
    backbone = EfficientViT(width=[3, 32, 64, 128, 256, 512], depth=[1, 1, 1, 6, 6])
    neck = Neck(in_channels=[512, 256, 128], n=8)
    model = EfficientViTSAM(backbone=backbone,
                            neck=neck,
                            prompt_encoder=PromptEncoder(embed_dim=256,
                                                         image_embedding_size=(64, 64),
                                                         input_image_size=(1024, 1024),
                                                         mask_in_chans=16),
                            mask_decoder=MaskDecoder(num_multi_mask_outputs=3,
                                                     transformer=TwoWayTransformer(depth=2,
                                                                                   embedding_dim=256,
                                                                                   mlp_dim=2048,
                                                                                   num_heads=8),
                                                     transformer_dim=256,
                                                     iou_head_depth=3,
                                                     iou_head_hidden_dim=256),
                            image_size=(1024, image_size))
    state_dict = torch.load('./weights/sam_l1.pt')['state_dict']
    model.load_state_dict(state_dict)
    return model


def build_sam_l2(image_size=512):
    if not exists('./weights'):
        makedirs('./weights')
    if not exists('./weights/sam_l2.pt'):
        url = 'https://github.com/jahongir7174/EfficientViTSAM/releases/download/v0.0.1/sam_l2.pt'
        torch.hub.download_url_to_file(url=url, dst='./weights/sam_l2.pt')
    backbone = EfficientViT(width=[3, 32, 64, 128, 256, 512], depth=[1, 2, 2, 8, 8])
    neck = Neck(in_channels=[512, 256, 128], n=12)
    model = EfficientViTSAM(backbone=backbone,
                            neck=neck,
                            prompt_encoder=PromptEncoder(embed_dim=256,
                                                         image_embedding_size=(64, 64),
                                                         input_image_size=(1024, 1024),
                                                         mask_in_chans=16),
                            mask_decoder=MaskDecoder(num_multi_mask_outputs=3,
                                                     transformer=TwoWayTransformer(depth=2,
                                                                                   embedding_dim=256,
                                                                                   mlp_dim=2048,
                                                                                   num_heads=8),
                                                     transformer_dim=256,
                                                     iou_head_depth=3,
                                                     iou_head_hidden_dim=256),
                            image_size=(1024, image_size))
    state_dict = torch.load('./weights/sam_l2.pt')['state_dict']
    model.load_state_dict(state_dict)
    return model
