# Video Swin Transformer

Keras Implementation of Video Swin Transformers. The official implementation is [here](https://github.com/SwinTransformer/Video-Swin-Transformer) in PyTorch based on [mmaction2](https://github.com/open-mmlab/mmaction2).

![](./assets/teaser.png)

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

