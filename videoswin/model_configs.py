MODEL_CONFIGS = {
    "VideoSwinB_K400_IN1K_P244_W877_32x224": {
        "patch_size": (2, 4, 4),
        "drop_path_rate": 0.3,
        "window_size": (8, 7, 7),
        "num_classes": 400,
    },
    "VideoSwinB_K400_IN22K_P244_W877_32x224": {
        "patch_size": (2, 4, 4),
        "drop_path_rate": 0.2,
        "window_size": (8, 7, 7),
        "num_classes": 400,
    },
    "VideoSwinB_K600_IN22K_P244_W877_32x224": {
        "patch_size": (2, 4, 4),
        "drop_path_rate": 0.2,
        "window_size": (8, 7, 7),
        "num_classes": 600,
    },
    "VideoSwinB_SSV2_K400_P244_W1677_32x224": {
        "patch_size": (2, 4, 4),
        "window_size": (16, 7, 7),
        "drop_path_rate": 0.4,
        "num_classes": 174,
    },
    "VideoSwinS_K400_IN1K_P244_W877_32x224": {
        "patch_size": (2, 4, 4),
        "drop_path_rate": 0.1,
        "window_size": (8, 7, 7),
        "num_classes": 400,
    },
    "VideoSwinT_K400_IN1K_P244_W877_32x224": {
        "patch_size": (2, 4, 4),
        "drop_path_rate": 0.1,
        "window_size": (8, 7, 7),
        "num_classes": 400,
    },
}
