
# VideoSwin Model Zoo

## Note

- `#Frame = #input_frame x #clip x #crop`. The frame interval is `2` to evaluate on benchmark dataset. 
- `#input_frame` means how many frames are input for model during the test phase.
- `#crop` means spatial crops (e.g., 3 for left/right/center crop).
- `#clip` means temporal clips (e.g., 5 means repeted temporal sampling five clips with different start indices).

### Kinetics 400

In the training phase, the video swin mdoels are initialized with the pretrained weights of image swin models. In that case, `IN` referes to **ImageNet**.

| Backbone |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-T  | IN-1K  | 32x4x3 | 78.8  |  93.6  |   [SavedModel](https://github.com/innat/VideoSwin/releases/download/v1.1/TFVideoSwinT_K400_IN1K_P244_W877_32x224.zip)/[h5](https://github.com/innat/VideoSwin/releases/download/v1.0/TFVideoSwinT_K400_IN1K_P244_W877_32x224.h5)  | 
|  Swin-S  | IN-1K  | 32x4x3 | 80.6  |  94.5  |   [SavedModel](https://github.com/innat/VideoSwin/releases/download/v1.1/TFVideoSwinS_K400_IN1K_P244_W877_32x224.zip)/[h5](https://github.com/innat/VideoSwin/releases/download/v1.0/TFVideoSwinS_K400_IN1K_P244_W877_32x224.h5)  |
|  Swin-B  | IN-1K  | 32x4x3 | 80.6  |  94.6  |   [SavedModel](https://github.com/innat/VideoSwin/releases/download/v1.1/TFVideoSwinB_K400_IN1K_P244_W877_32x224.zip)/[h5](https://github.com/innat/VideoSwin/releases/download/v1.0/TFVideoSwinB_K400_IN1K_P244_W877_32x224.h5)  | 
|  Swin-B  | IN-22K | 32x4x3 | 82.7  |  95.5  |   [SavedModel](https://github.com/innat/VideoSwin/releases/download/v1.1/TFVideoSwinB_K400_IN22K_P244_W877_32x224.zip)/[h5](https://github.com/innat/VideoSwin/releases/download/v1.0/TFVideoSwinB_K400_IN22K_P244_W877_32x224.h5)   | 

### Kinetics 600

| Backbone |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | IN-22K | 32x4x3 | 84.0  |  96.5  |   [SavedModel](https://github.com/innat/VideoSwin/releases/download/v1.1/TFVideoSwinB_K600_IN22K_P244_W877_32x224.zip)/[h5](https://github.com/innat/VideoSwin/releases/download/v1.0/TFVideoSwinB_K600_IN22K_P244_W877_32x224.h5)  | 

### Something-Something V2

| Backbone |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | Kinetics 400 | 32x1x3 | 69.6  |  92.7  |  [SavedModel](https://github.com/innat/VideoSwin/releases/download/v1.1/TFVideoSwinB_SSV2_K400_P244_W1677_32x224.zip)/[h5](https://github.com/innat/VideoSwin/releases/download/v1.0/TFVideoSwinB_SSV2_K400_P244_W1677_32x224.h5)  | 


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

