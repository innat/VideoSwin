
# VideoSwin Model Zoo

### Kinetics 400

| Backbone |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-T  | IN-1K  | 32x4x3 | 78.8  |  93.6  |   [SavedModel]()/[h5]()  | 
|  Swin-S  | IN-1K  | 32x4x3 | 80.6  |  94.5  |   [SavedModel]()/[h5]()  |
|  Swin-B  | IN-1K  | 32x4x3 | 80.6  |  94.6  |   [SavedModel]()/[h5]()  | 
|  Swin-B  | IN-22K | 32x4x3 | 82.7  |  95.5  |   [SavedModel]()/[h5]()   | 

### Kinetics 600

| Backbone |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | IN-22K | 32x4x3 | 84.0  |  96.5  |   [SavedModel]()/[h5]()  | 

### Something-Something V2

| Backbone |  Pretrain  | #Frame | Top-1 | Top-5 | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | 
|  Swin-B  | Kinetics 400 | 32x1x3 | 69.6  |  92.7  |  [SavedModel]()/[h5]()  | 


## Note

- `#Frame = #input_frame x #clip x #crop`. The frame interval is `2` to evaluate on benchmark dataset. 
- `#input_frame` means how many frames are input for model during the test phase.
- `#crop` means spatial crops (e.g., 3 for left/right/center crop).
- `#clip` means temporal clips (e.g., 5 means repeted temporal sampling five clips with different start indices).#