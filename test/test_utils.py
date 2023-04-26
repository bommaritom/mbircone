import numpy as np
import os
import mbircone.cone3D as cone3D
import matplotlib.pyplot as plt
from scipy import signal as sgn
from scipy.ndimage import gaussian_filter, zoom

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')


def block_average(image, k, ell):
    """ Return a block-averaged version of image
        with k by ell kernel size

    Args:
        image (float, ndarray): 3D image array
        k (int): Size of blocks along axis 0
        ell (int): Size of blocks along axis 1

    Returns:
        (float, ndarray): Result of block averaging over blocks of size k x ell
    """
    m, n = image.shape

    # zero-padding to be zero modulo k and ell
    # the array will become size (m+k-1)//k by (n+ell-1)//ell
    image_pad = np.pad(image, ((0, (-m) % k), (0, (-n) % ell)), 'constant')

    # turn into an array of k by ell blocks
    image_reshape = image_pad.reshape((m + k - 1) // k, k, (n + ell - 1) // ell, ell).swapaxes(1, 2)

    # get average of the blocks
    image_block_avg = np.mean(image_reshape, axis=2)
    image_block_avg = np.mean(image_block_avg, axis=2)

    return image_block_avg


def block_average_3D(image, k, ell, s):
    """ Return a block-averaged version of image
        with k by ell by s kernel size

    Args:
        image (float, ndarray): 3D image array
        k (int) : Size of blocks along axis 0
        ell (int) : Size of blocks along axis 1
        s (int) : Size of blocks along axis 2

    Returns:
        (float, ndarray): Result of block averaging over blocks of size k x ell x s
    """
    m, n, p = image.shape

    # zero-padding to be zero modulo k, ell, and s
    # the array will become size (m+k-1)//k by (n+ell-1)//ell by (p+s-1)//s
    image_pad = np.pad(image, ((0, (-m) % k), (0, (-n) % ell), (0, (-p) % s)), 'constant')

    # turn into an array of k by ell by s blocks
    image_reshape = image_pad.reshape((m + k - 1) // k, k,
                                      (n + ell - 1) // ell, ell,
                                      (p + s - 1) // s, s).swapaxes(1, 2).swapaxes(2, 4).swapaxes(4, 5)

    # get average of the blocks
    image_block_avg = np.mean(image_reshape, axis=3)
    image_block_avg = np.mean(image_block_avg, axis=3)
    image_block_avg = np.mean(image_block_avg, axis=3)

    return image_block_avg


def block_average_sino(sino):
    """ Perform view-wise block-averaging of the inputted sinogram.

    Args:
        sino (float, ndarray): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).

    Returns:
        (float, ndarray) 3D block-averaged sinogram data
        with shape ( num_views, (num_det_rows+1)//2, (num_det_channels+1)//2 )

    """
    num_views, num_det_rows, num_det_channels = np.shape(sino)

    # perform block-averaging
    sino_block_average = np.zeros((num_views, (num_det_rows + 1) // 2, (num_det_channels + 1) // 2))
    for view in range(num_views):
        sino_block_average[view] = block_average(np.copy(sino)[view], 2, 2)

    return sino_block_average


def zoom_sino(sino):
    """ Apply the scipy zoom method to each view in the sinogram to rescale by factor of 2.

    Args:
        sino (float, ndarray): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).

    Returns:
        (float, ndarray) Low-resolution sinogram attained using scipy.ndimage.zoom method,
        of shape (num_views // 2, num_det_rows // 2, num_det_channels // 2).


    """
    num_views, num_det_rows, num_det_channels = np.shape(sino)

    # perform block-averaging
    sino_zoom = np.zeros((num_views, (num_det_rows + 1) // 2, (num_det_channels + 1) // 2))
    for view in range(num_views):
        sino_zoom[view] = zoom(np.copy(sino)[view], 0.5)

    return sino_zoom


def apply_gaussian_filter_to_sino(sino, sigma=1):
    """ Apply a Gaussian filter (scipy.ndimage.gaussian_filter) to each view in sinogram.

    Args:
        sino (float, ndarray): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).

    Returns:
        (float, ndarray) 3D filtered sinogram data with shape (num_views, num_det_rows, num_det_channels).
    """
    num_views, num_det_rows, num_det_channels = np.shape(sino)

    sino_filtered = np.zeros((num_views, num_det_rows, num_det_channels))
    for view in range(num_views):
        sino_filtered[view] = gaussian_filter(sino[view], sigma=sigma)

    return sino_filtered


def plot_image(img, title=None, filename=None, vmin=None, vmax=None):
    """
    Function to display and save a 2D array as an image.

    Args:
        img: 2D numpy array to display
        title: Title of plot image
        filename: A path to save plot image
        vmin: Value mapped to black
        vmax: Value mapped to white
    """

    plt.ion()
    fig = plt.figure()
    imgplot = plt.imshow(img, vmin=vmin, vmax=vmax, interpolation='none')
    plt.title(label=title)
    imgplot.set_cmap('gray')
    plt.colorbar()
    if filename != None:
        try:
            plt.savefig(filename)
        except:
            print("plot_image() Warning: Can't write to file {}".format(filename))

