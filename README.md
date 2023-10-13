# Video Swin Transformer

![](./assets/teaser.png)

[![arXiv](https://img.shields.io/badge/arXiv-2203.12602-darkred)](https://arxiv.org/abs/2203.12602) [![keras-2.12.](https://img.shields.io/badge/keras-2.12-darkred)]([?](https://img.shields.io/badge/keras-2.12-darkred)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BFisOW2yzdvDEBN_0P3M41vQCwF6dTWR?usp=sharing) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/innat/VideoMAE) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Hub-yellow.svg)](https://huggingface.co/innat/videomae)


Keras Implementation of Video Swin Transformers. The official implementation is [here](https://github.com/SwinTransformer/Video-Swin-Transformer) in PyTorch based on [mmaction2](https://github.com/open-mmlab/mmaction2).

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

### Kinetics 400

| Backbone |  Pretrain    | acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Swin-T  | IN-1K |  78.8  |  93.6  |   28M   |  87.9G  |  ?  | ? |
|  Swin-S  | IN-1K |  80.6  |  94.5  |   50M   |  165.9G  |  ?  | ? |
|  Swin-B  | IN-1K |  80.6  |  94.6  |   88M   |  281.6G  |  ?  | ? |
|  Swin-B  | IN-22K | 82.7  |  95.5  |   88M   |  281.6G  |  ?  | ? |

### Kinetics 600

| Backbone |  Pretrain   | acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: | :---: |  ::---: | :---: | :---: | :---: | :---: |
|  Swin-B  | ImageNet-22K | 84.0  |  96.5  |   88M   |  281.6G  |  ?  | ? |

### Something-Something V2

| Backbone |  Pretrain   |  acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: |  :---: |  :---: | :---: | :---: | :---: | :---: |
|  Swin-B  | Kinetics 400 |  69.6  |  92.7  |   89M   |  320.6G  |  ?  | ? |

