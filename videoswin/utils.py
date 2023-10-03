import numpy as np
import tensorflow as tf

def tf_window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
        
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    
    input_shape = tf.shape(x)
    B, D, H, W, C = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    )
    
    x = tf.reshape(
        x, 
        [
            B,
            D // window_size[0], window_size[0], 
            H // window_size[1], window_size[1], 
            W // window_size[2], window_size[2], 
            C
        ]
    )
    
    x = tf.transpose(x, perm=[0,1,3,5,2,4,6,7])
    windows = tf.reshape(x, [-1, window_size[0]*window_size[1]*window_size[2], C])      
    
    return windows


def tf_window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = tf.reshape(
        windows,
        [
            B, 
            D // window_size[0], 
            H // window_size[1], 
            W // window_size[2], 
            window_size[0], 
            window_size[1], 
            window_size[2], 
            -1
        ]
    )
    x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3, 6, 7])
    x = tf.reshape(x, [B, D, H, W, -1])
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
    

def tf_compute_mask(D, H, W, window_size, shift_size):
    img_mask = np.zeros((1, D, H, W, 1))
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt = cnt + 1
    mask_windows = tf_window_partition(img_mask, window_size) 
    mask_windows = tf.squeeze(mask_windows, axis = -1)
    attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
    attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
    attn_mask = tf.where(attn_mask == 0, 0.0 , attn_mask)
    return attn_mask
