import numpy as np
from keras import ops

def window_partition(x, window_size):
    """
    Args:
        x: (batch_size, depth, height, width, channel)
        window_size (tuple[int]): window size
        
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    
    input_shape = ops.shape(x)
    batch_size, depth, height, width, channel = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    )
    
    x = ops.reshape(
        x, 
        [
            batch_size,
            depth // window_size[0], window_size[0], 
            height // window_size[1], window_size[1], 
            width // window_size[2], window_size[2], 
            channel
        ]
    )
    
    x = ops.transpose(x, [0,1,3,5,2,4,6,7])
    windows = ops.reshape(x, [-1, window_size[0]*window_size[1]*window_size[2], C])      
    
    return windows


def window_reverse(windows, window_size, batch_size, depth, height, width):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        height (int): Height of image
        width (int): Width of image

    Returns:
        x: (batch_size, depth, height, width, channel)
    """
    x = ops.reshape(
        windows,
        [
            batch_size, 
            depth // window_size[0], 
            height // window_size[1], 
            width // window_size[2], 
            window_size[0], 
            window_size[1], 
            window_size[2], 
            -1
        ]
    )
    x = ops.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
    x = ops.reshape(x, [batch_size, depth, height, width, -1])
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    
    if shift_size is not None:
        use_shift_size = list(shift_size)
        
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
                
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
    

def compute_mask(depth, height, width, window_size, shift_size):
    img_mask = np.zeros((1, depth, height, width, 1))
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt = cnt + 1
    mask_windows = window_partition(img_mask, window_size) 
    mask_windows = ops.squeeze(mask_windows, axis = -1)
    attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(mask_windows, axis=2)
    attn_mask = ops.where(attn_mask != 0, -100.0, attn_mask)
    attn_mask = ops.where(attn_mask == 0, 0.0 , attn_mask)
    return attn_mask
