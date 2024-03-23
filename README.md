# Video Swin Transformer

![](./assets/teaser.png)

[![Palestine](https://img.shields.io/badge/Free-Palestine-white?labelColor=green)](https://twitter.com/search?q=%23FreePalestine&src=typed_query)

[![arXiv](https://img.shields.io/badge/arXiv-2106.13230-darkred)](https://arxiv.org/abs/2106.13230) [![keras-3](https://img.shields.io/badge/keras-3-darkred
)]([?](https://img.shields.io/badge/keras-2.12-darkred)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q7A700MEI10UomikqjQJANWyFZktJCT-?usp=sharing) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/innat/VideoSwin) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Hub-yellow.svg)](https://huggingface.co/innat/videoswin)


VideoSwin is a pure transformer based video modeling algorithm, attained top accuracy on the major video recognition benchmarks. In this model, the author advocates an inductive bias of locality in video transformers, which leads to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization. The locality of the proposed video architecture is realized by adapting the [**Swin Transformer**](https://arxiv.org/abs/2103.14030) designed for the image domain, while continuing to leverage the power of pre-trained image models.

This is a unofficial `Keras 3` implementation of [Video Swin transformers](https://arxiv.org/abs/2106.13230). The official `PyTorch` implementation is [here](https://github.com/SwinTransformer/Video-Swin-Transformer) based on [mmaction2](https://github.com/open-mmlab/mmaction2). The official PyTorch weight has been converted to `Keras 3` compatible. This implementaiton supports to run the model on multiple backend, i.e. TensorFlow, PyTorch, and Jax.


# Install 

```python
git clone https://github.com/innat/VideoSwin.git
cd VideoSwin
pip install -e . 
```

# Checkpoints

The **VideoSwin** checkpoints are available in both `.weights.h5`, and `.keras` formats. The variants of this models are `tiny`, `small`, and `base`. Check [model zoo](https://github.com/innat/VideoSwin/blob/main/MODEL_ZOO.md) page to know details of it. 


**Inference**

```python
from videoswin import VideoSwinT

>>> import  os
>>> import torch
>>> os.environ["KERAS_BACKEND"] = "torch"
>>> from videoswin import VideoSwinT

>>> model = VideoSwinT(
    num_classes=400,
    include_rescaling=False,
    activation=None
)
>>> _ = model(torch.ones((1, 32, 224, 224, 3)))
>>> model.load_weights('model.weights.h5')

>>> container = read_video('sample.mp4')
>>> frames = frame_sampling(container, num_frames=32)
>>> y_pred = model(frames)
>>> y_pred.shape
TensorShape([1, 400])

>>> probabilities = torch.nn.functional.softmax(y_pred).detach().numpy()
>>> probabilities = probabilities.squeeze(0)
>>> confidences = {
    label_map_inv[i]: float(probabilities[i]) \
    for i in np.argsort(probabilities)[::-1]
}
>>> confidences
```
A classification results on a sample from [Kinetics-400](https://www.deepmind.com/open-source/kinetics).

| Video | Top-5 |
|:---:|:---|
| ![](./assets/view1.gif) | <pre>{<br>    'playing_cello': 0.9941741824150085,<br>    'playing_violin': 0.0016851733671501279,<br>    'playing_recorder': 0.0011555481469258666,<br>    'playing_clarinet': 0.0009695519111119211,<br>    'playing_harp': 0.0007713600643910468<br>}</pre> |


**Fine Tune**

Each videoswin checkpoints returns `logits`. We can just add a custom classifier on top of it. For example:

```python
# import pretrained model, i.e.
video_swin = keras.models.load_model(
    'TFVideoSwinB_SSV2_K400_P244_W1677_32x224', compile=False
    )
video_swin.trainable = False

# downstream model
model = keras.Sequential([
    video_swin,
    layers.Dense(
        len(class_folders), dtype='float32', activation=None
    )
])
model.compile(...)
model.fit(...)
model.predict(...)
```

## Model Zoo

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
- [ ] Publish on TF-Hub.
- [ ] Support `Keras V3` to support multi-framework backend.

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
