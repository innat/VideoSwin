# Video Swin Transformer

Keras Implementation of Video Swin Transformers. The official implementation is [here](https://github.com/SwinTransformer/Video-Swin-Transformer) in PyTorch based on [mmaction2](https://github.com/open-mmlab/mmaction2).

![](./assets/teaser.png)

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

