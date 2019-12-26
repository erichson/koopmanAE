import matplotlib
import matplotlib.cm

import numpy as np
import torch

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = torch.min(value) if vmin is None else vmin
    vmax = torch.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = torch.squeeze(value)

    # quantize
    indices = torch.round(value * 255).long()

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    # colors = tf.constant(cm.colors, dtype=tf.float32)
    colors = cm(np.arange(256))[:, :3]
    colors = torch.from_numpy(colors)
    value = colors[indices]

    return value