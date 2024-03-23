
# Video Swin Transformer Model Zoo

Video Swin in `keras` can be used with multiple backends, i.e. `tensorflow`, `torch`, and `jax`. The input shape are expected to be `channel_last`, i.e. `(depth, height, width, channel)`.

## Note

While evaluating the video model for classification task, multiple clips from a video are sampled. This process also involves multiple crops on the sample. 

- `#Frame = #input_frame x #clip x #crop`. The frame interval is `2` to evaluate on benchmark dataset. 
- `#input_frame` means how many frames are input for model during the test phase. For video swin, it is `32`.
- `#crop` means spatial crops (e.g., 3 for left/right/center crop).
- `#clip` means temporal clips (e.g., 5 means repeted temporal sampling five clips with different start indices).


# Checkpoints

In the training phase, the video swin mdoels are initialized with the pretrained weights of image swin models. In that case, `IN` referes to **ImageNet**. In the following, the `keras` checkpoints are the complete model, so `keras.saving.load_model` API can be used. In contrast, the `h5` checkpoints are the only weight file.

### Kinetics 400

| Model |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints | config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-T  | IN-1K  | 32x4x3 | 78.8  |  93.6  |   [keras]()/[h5]() | [swin-t](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py)  |
|  Swin-S  | IN-1K  | 32x4x3 | 80.6  |  94.5  |   [keras]()/[h5]() | [swin-s](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py)  |
|  Swin-B  | IN-1K  | 32x4x3 | 80.6  |  94.6  |   [keras]()/[h5]() | [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py)  |
|  Swin-B  | IN-22K | 32x4x3 | 82.7  |  95.5  |   [keras]()/[h5]() | [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py)  |

### Kinetics 600

| Model |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints | config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | IN-22K | 32x4x3 | 84.0  |  96.5  |   [keras]()/[h5]()  |  [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py)  | 

### Something-Something V2

| Model |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints | config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | Kinetics 400 | 32x1x3 | 69.6  |  92.7  |  [keras]()/[h5]()  |  [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py)  |


## Weight Comparison

The `torch` videoswin model can be loaded from the official [repo](https://github.com/SwinTransformer/Video-Swin-Transformer). Following are some quick test of both implementation showing logit matching.

```python
input = np.random.rand(4, 32, 224, 224, 3).astype('float32')
inputs = torch.tensor(input)
inputs = torch.einsum('nthwc->ncthw', inputs)
# inputs.shape: torch.Size([4, 3, 32, 224, 224])

# torch model
model_pt.eval()
x = model_torch(inputs.float())
x = x.detach().numpy()
x.shape # (4, 174) (Sth-Sth dataset)

# keras model
y = model_keras(input, training=False)
y = y.numpy()
y.shape # (4, 174) (Sth-Sth dataset)

np.testing.assert_allclose(x, y, 1e-4, 1e-4)
np.testing.assert_allclose(x, y, 1e-5, 1e-5)
# OK
```
