
# Video Swin Transformer Model Zoo

Video Swin in `keras` can be used with multiple backends, i.e. `tensorflow`, `torch`, and `jax`. The input shape are expected to be `channel_last`, i.e. `(depth, height, width, channel)`. 

**Note**: While evaluating the video model for classification task, multiple clips from a video are sampled. And additionally, this process also involves multiple crops on the sample. So, while evaluating on benchmark dataset, we should consider this current standard. Check the official [config](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/db018fb8896251711791386bbd2127562fd8d6a6/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py#L45-L61).

- `#Frame = #input_frame x #clip x #crop`. The frame interval is `2` to evaluate on benchmark dataset. 
- `#input_frame` means how many frames are input for model during the test phase. For video swin, it is `32`.
- `#crop` means spatial crops (e.g., 3 for left/right/center crop).
- `#clip` means temporal clips (e.g., 4 means repeted temporal sampling five clips with different start indices).


# Checkpoints

In the training phase, the video swin mdoels are initialized with the pretrained weights of image swin models. In the following table, `IN` referes to **ImageNet**. By default, the video swin is trained with input shape of `32, 224, 224, 3`. 

### Kinetics 400

| Model |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints | config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-T  | IN-1K  | 32x4x3 | 78.8  |  93.6  |   [h5](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_tiny_kinetics400_classifier.weights.h5) / [h5-no-top](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_tiny_kinetics400.weights.h5) | [swin-t](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py)  |
|  Swin-S  | IN-1K  | 32x4x3 | 80.6  |  94.5  |   [h5](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_small_kinetics400_classifier.weights.h5) / [h5-no-top](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_small_kinetics400.weights.h5) | [swin-s](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py)  |
|  Swin-B  | IN-1K  | 32x4x3 | 80.6  |  94.6  |  [h5](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_base_kinetics400_classifier.weights.h5) / [h5-no-top](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_base_kinetics400.weights.h5) | [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py)  |
|  Swin-B  | IN-22K | 32x4x3 | 82.7  |  95.5  |   [h5](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_base_kinetics400_imagenet22k_classifier.weights.h5) / [h5-no-top](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_base_kinetics400_imagenet22k.weights.h5) | [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py)  |

### Kinetics 600

| Model |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints | config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | IN-22K | 32x4x3 | 84.0  |  96.5  |   [h5](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_base_kinetics600_imagenet22k_classifier.weights.h5) / [h5-no-top](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_base_kinetics600_imagenet22k.weights.h5)  |  [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py)  | 

### Something-Something V2

| Model |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints | config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | Kinetics 400 | 32x1x3 | 69.6  |  92.7  |  [h5](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_base_something_something_v2_classifier.weights.h5) / [h5-no-top](https://github.com/innat/VideoSwin/releases/download/v2.0/videoswin_base_something_something_v2.weights.h5) |  [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py)  |

