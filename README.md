# Video Swin Transformer

Keras Implementation of Video Swin Transformers. The official implementation is [here](https://github.com/SwinTransformer/Video-Swin-Transformer) in PyTorch based on [mmaction2](https://github.com/open-mmlab/mmaction2).

![](./assets/teaser.png)

```python
def tf_video_swin_tiny(**kwargs):
    model = TFSwinTransformer3D(
        patch_size=(2,4,4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8,7,7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=partial(
            layers.LayerNormalization, epsilon=1e-05
        ),
        patch_norm=True,
        in_channels=768,
        **kwargs
    )
    return model


model = tf_video_swin_tiny(num_classes=400)
model_tf.load_weights('TFVideoSwinT_K400_IN1K_P244_W877_32x224.h5')

y_pred = model_tf(tf.ones(shape=(1, 32, 224, 224, 3)))
y_pred.shape
TensorShape([1, 400])
```

## Results and Models

### Kinetics 400

| Backbone |  Pretrain    | spatial crop | acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Swin-T  | IN-1K |      224      |  78.8  |  93.6  |   28M   |  87.9G  |  ?  | ? |
|  Swin-S  | IN-1K |      224      |  80.6  |  94.5  |   50M   |  165.9G  |  ?  | ? |
|  Swin-B  | IN-1K |      224      |  80.6  |  94.6  |   88M   |  281.6G  |  ?  | ? |
|  Swin-B  | IN-22K |     224      |  82.7  |  95.5  |   88M   |  281.6G  |  ?  | ? |

### Kinetics 600

| Backbone |  Pretrain   |  spatial crop | acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
|  Swin-B  | ImageNet-22K |      224      |  84.0  |  96.5  |   88M   |  281.6G  |  ?  | ? |

### Something-Something V2

| Backbone |  Pretrain   |  spatial crop | acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Swin-B  | Kinetics 400 |    224      |  69.6  |  92.7  |   89M   |  320.6G  |  ?  | ? |

