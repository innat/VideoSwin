# Video Swin Transformer

![](./assets/teaser.png)


[![arXiv](https://img.shields.io/badge/arXiv-2106.13230-darkred)](https://arxiv.org/abs/2106.13230) [![keras-2.12.](https://img.shields.io/badge/keras-2.12-darkred)]([?](https://img.shields.io/badge/keras-2.12-darkred)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](?) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](?) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Hub-yellow.svg)](?)


Keras implementation of Video Swin transformers. The official implementation is [here](https://github.com/SwinTransformer/Video-Swin-Transformer) in PyTorch based on [mmaction2](https://github.com/open-mmlab/mmaction2).

## News

- 

# Install 

```python
git clone https://github.com/innat/VideoSwin.git
cd VideoSwin
pip install -e . 
```

# Usage

```python
from videomae import VideoSwinT

>>> model = VideoSwinT(num_classes=400)
>>> model.load_weights('TFVideoSwinT_K400_IN1K_P244_W877_32x224.h5')
>>> container = read_video('sample.mp4')
>>> frames = frame_sampling(container, num_frames=32)
>>> y = model(frames)
>>> y.shape
TensorShape([1, 400])

>>> probabilities = tf.nn.softmax(y_pred_tf)
>>> probabilities = probabilities.numpy().squeeze(0)
>>> confidences = {
    label_map_inv[i]: float(probabilities[i]) \
    for i in np.argsort(probabilities)[::-1]
}
>>> confidences

```

## Results and Models

The 3D swin-video checkpoints are listed in [`MODEL_ZOO.md`](MODEL_ZOO.md). Following are some hightlights.

### Kinetics 400

In the training phase, the video swin mdoels are initialized with the pretrained weights of image swin models. In that case, `IN` referes to **ImageNet**.

| Backbone |  Pretrain  | Top-1 | Top-5 | #params | FLOPs | config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-T  | IN-1K |  78.8  |  93.6  |   28M   |  ?   |  [swin-t](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py)  |
|  Swin-S  | IN-1K |  80.6  |  94.5  |   50M   |  ?  |  [swin-s](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py)  |
|  Swin-B  | IN-1K |  80.6  |  94.6  |   88M   |  ?  |  [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py)  |
|  Swin-B  | IN-22K | 82.7  |  95.5  |   88M   |  ?  |  [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py)  |

### Kinetics 600

| Backbone |  Pretrain   | Top-1 | Top-5 | #params | FLOPs | config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | IN-22K | 84.0  |  96.5  |   88M   |  ?  |  [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py)  | 

### Something-Something V2

| Backbone |  Pretrain   |  Top-1 | Top-5 | #params | FLOPs | config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | Kinetics 400 |  69.6  |  92.7  |   89M   |  ?  |  [swin-b](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py)  |


# TODO
- [x] Custom fine-tuning code.
- [] Publish on TF-Hub.
- [] Support `Keras V3` to support multi-framework backend.

##  Citation

If you use this videoswin implementation in your research, please cite it using the metadata from our `CITATION.cff` file.

```swift
@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}
```