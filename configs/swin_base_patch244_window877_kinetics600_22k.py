_base_ = "swin_base_patch244_window877_kinetics400_22k.py"

data_root = 'data/kinetics600/train'
data_root_val = 'data/kinetics600/val'
ann_file_train = 'data/kinetics600/kinetics600_train_list.txt'
ann_file_val = 'data/kinetics600/kinetics600_val_list.txt'
ann_file_test = 'data/kinetics600/kinetics600_val_list.txt'


model=dict(cls_head=dict(num_classes=600))