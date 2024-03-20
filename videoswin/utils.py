import numpy as np
from keras import ops


def window_partition(x, window_size):
    """Partitions a video tensor into non-overlapping windows of a specified size.

    Args:
        x: A tensor with shape (B, D, H, W, C), where:
            - B: Batch size
            - D: Number of frames (depth) in the video
            - H: Height of the video frames
            - W: Width of the video frames
            - C: Number of channels in the video (e.g., RGB for color)
        window_size: A tuple of ints of size 3 representing the window size
            along each dimension (depth, height, width).

    Returns:
        A tensor with shape (num_windows * B, window_size[0], window_size[1], window_size[2], C),
        where each window from the video is a sub-tensor containing the specified
        number of frames and the corresponding spatial window.
    """  # noqa: E501

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
            depth // window_size[0],
            window_size[0],
            height // window_size[1],
            window_size[1],
            width // window_size[2],
            window_size[2],
            channel,
        ],
    )

    x = ops.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    windows = ops.reshape(
        x, [-1, window_size[0] * window_size[1] * window_size[2], channel]
    )

    return windows


def window_reverse(windows, window_size, batch_size, depth, height, width):
    """Reconstructs the original video tensor from its partitioned windows.

    This function assumes the windows were created using the `window_partition` function
    with the same `window_size`.

    Args:
        windows: A tensor with shape (num_windows * batch_size, window_size[0],
            window_size[1], window_size[2], channels), where:
            - num_windows: Number of windows created during partitioning
            - channels: Number of channels in the video (same as in `window_partition`)
        window_size: A tuple of ints of size 3 representing the window size used
            during partitioning (same as in `window_partition`).
        batch_size: Batch size of the original video tensor (same as in `window_partition`).
        depth: Number of frames (depth) in the original video tensor (same as in `window_partition`).
        height: Height of the video frames in the original tensor (same as in `window_partition`).
        width: Width of the video frames in the original tensor (same as in `window_partition`).

    Returns:
        A tensor with shape (batch_size, depth, height, width, channels), representing the
        original video reconstructed from the provided windows.
    """  # noqa: E501
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
            -1,
        ],
    )
    x = ops.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
    x = ops.reshape(x, [batch_size, depth, height, width, -1])
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computes the appropriate window size and potentially shift size for Swin Transformer.

    This function implements the logic from the Swin Transformer paper by Ze Liu et al.
    (https://arxiv.org/abs/2103.14030) to determine suitable window sizes
    based on the input size and the provided base window size.

    Args:
        x_size: A tuple of ints of size 3 representing the input size (depth, height, width)
            of the data (e.g., video).
        window_size: A tuple of ints of size 3 representing the base window size
            (depth, height, width) to use for partitioning.
        shift_size: A tuple of ints of size 3 (optional) representing the window
            shifting size (depth, height, width) for shifted window processing
            used in Swin Transformer. If not provided, only window size is computed.

    Returns:
        A tuple or a pair of tuples:
            - If `shift_size` is None, returns a single tuple representing the adjusted
            window size that may be smaller than the provided `window_size` to ensure
            it doesn't exceed the input size along any dimension.
            - If `shift_size` is provided, returns a pair of tuples. The first tuple
            represents the adjusted window size, and the second tuple represents the
            adjusted shift size. The adjustments ensure both window size and shift size
            do not exceed the corresponding dimensions in the input data.
    """  # noqa: E501

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
    """Computes an attention mask for a sliding window self-attention mechanism
    used in Video Swin Transformer.

    This function creates a mask to indicate which windows can attend to each other
    during the self-attention operation. It considers non-overlapping and potentially
    shifted windows based on the provided window size and shift size.

    Args:
        depth (int): Depth (number of frames) of the input video.
        height (int): Height of the video frames.
        width (int): Width of the video frames.
        window_size (tuple[int]): Size of the sliding window in each dimension
            (depth, height, width).
        shift_size (tuple[int]): Size of the shifting step in each dimension
            (depth, height, width).

    Returns:
        A tensor of shape (batch_size, num_windows, num_windows), where:
            - batch_size: Assumed to be 1 in this function.
            - num_windows: Total number of windows covering the entire input based on
                the formula:
                    (depth - window_size[0]) // shift_size[0] + 1) *
                    (height - window_size[1]) // shift_size[1] + 1) *
                    (width - window_size[2]) // shift_size[2] + 1)
        Each element (attn_mask[i, j]) represents the attention weight between
        window i and window j. A value of -100.0 indicates high negative attention
        (preventing information flow), 0.0 indicates no mask effect.
    """  # noqa: E501

    img_mask = np.zeros((1, depth, height, width, 1))
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt = cnt + 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = ops.squeeze(mask_windows, axis=-1)
    attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(
        mask_windows, axis=2
    )
    attn_mask = ops.where(attn_mask != 0, -100.0, attn_mask)
    attn_mask = ops.where(attn_mask == 0, 0.0, attn_mask)
    return attn_mask
