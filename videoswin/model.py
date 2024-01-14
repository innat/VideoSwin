import os
import warnings
from functools import partial

import numpy as np

warnings.simplefilter(action="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import List, Optional, Tuple, Type, Union

import keras
from keras import layers

from videoswin.blocks import BasicLayer
from videoswin.layers import PatchEmbed3D, PatchMerging


@keras.utils.register_keras_serializable(package="swin.transformer.3d")
class SwinTransformer3D(keras.Model):
    """Swin Transformer backbone.
        A Keras impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: LayerNormalization.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = (4, 4, 4),
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: List[int] = (2, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: Type[layers.Layer] = layers.LayerNormalization,
        patch_norm: bool = False,
        num_classes: int = 400,
        **kwargs,
    ) -> keras.Model:
        super().__init__(**kwargs)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate = attn_drop_rate
        self.depths = depths
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.num_classes = num_classes

    def build(self, input_shape):
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None,
            name="PatchEmbed3D",
        )
        self.pos_drop = layers.Dropout(self.drop_rate, name="pos_drop")

        # stochastic depth
        dpr = np.linspace(0.0, self.drop_path_rate, sum(self.depths)).tolist()

        # build layers
        self.basic_layers = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop_rate=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[
                    sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                ],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                name=f"BasicLayer{i_layer+1}",
            )
            self.basic_layers.append(layer)

        self.norm = self.norm_layer(axis=-1, epsilon=1e-05, name="norm")
        self.avg_pool3d = layers.GlobalAveragePooling3D()
        self.head = layers.Dense(
            self.num_classes, use_bias=True, name="head", dtype="float32"
        )
        self.build_shape = input_shape[1:]

    def call(self, x, return_attention_maps=False, training=None):
        # tensor embeddings
        x = self.patch_embed(x)
        x = self.pos_drop(x, training=training)

        # video-swin block computation
        attention_maps_dict = {}
        for layer in self.basic_layers:
            if return_attention_maps:
                x, attention_maps = layer(
                    x, return_attention_maps=return_attention_maps, training=training
                )
                attention_maps_dict[f"{layer.name.lower()}_attention_maps"] = attention_maps
            else:
                x = layer(x, training=training)

        # head branch
        x = self.norm(x)
        x = self.avg_pool3d(x)
        x = self.head(x)

        if return_attention_maps:
            return x, attention_maps_dict

        return x

    def build_graph(self):
        x = keras.Input(shape=self.build_shape, name="input_graph")
        return keras.Model(inputs=[x], outputs=self.call(x))
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "patch_norm": self.patch_norm,
                "window_size": self.window_size,
                "patch_size": self.patch_size,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "num_classes": self.num_classes,
            }
        )
        return config


def VideoSwinT(num_classes, window_size=(8, 7, 7), drop_path_rate=0.2, **kwargs):
    model = SwinTransformer3D(
        num_classes=num_classes,
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
        patch_norm=True,
        **kwargs,
    )
    return model


def VideoSwinS(num_classes, window_size=(8, 7, 7), drop_path_rate=0.2, **kwargs):
    model = SwinTransformer3D(
        num_classes=num_classes,
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
        patch_norm=True,
        **kwargs,
    )
    return model


def VideoSwinB(num_classes, window_size=(8, 7, 7), drop_path_rate=0.2, **kwargs):
    model = SwinTransformer3D(
        num_classes=num_classes,
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
        patch_norm=True,
        **kwargs,
    )
    return model
