import numpy as np
import os
import mbircone.cone3D as cone3D

__lib_path = os.path.join(os.path.expanduser('~'), '.cache', 'mbircone')


def block_average(image, k, ell):
    """
    Return a block-averaged version of image
    with k by ell kernel size
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
    """
        Return a block-averaged version of image
        with k by ell kernel size
    """
    m, n, p = image.shape

    # zero-padding to be zero modulo k and ell
    # the array will become size (m+k-1)//k by (n+ell-1)//ell
    image_pad = np.pad(image, ((0, (-m) % k), (0, (-n) % ell), (0, (-p) % s)), 'constant')

    # turn into an array of k by ell blocks
    image_reshape = image_pad.reshape((m + k - 1) // k, k,
                                      (n + ell - 1) // ell, ell,
                                      (p + s - 1) // s, s).swapaxes(1, 2).swapaxes(2, 4).swapaxes(4, 5)

    # get average of the blocks
    image_block_avg = np.mean(image_reshape, axis=3)
    image_block_avg = np.mean(image_block_avg, axis=3)
    image_block_avg = np.mean(image_block_avg, axis=3)

    return image_block_avg


def half_res_sino(sino):
    """
    Applies a convolutional filter and then downsamples each view of the sinogram by block-averaging.
    """
    num_views, num_det_rows, num_det_channels = np.shape(sino)

    # perform block-averaging
    sino_block_average = np.zeros((num_views, (num_det_rows + 1) // 2, (num_det_channels + 1) // 2))
    for view in range(num_views):
        sino_block_average[view] = block_average(np.copy(sino)[view], 2, 2)

    return sino_block_average


def half_res_recon(sino, angles, dist_source_detector, magnification,
                   weights=None, weight_type='unweighted', init_image=0.0, prox_image=None,
                   num_image_rows=None, num_image_cols=None, num_image_slices=None,
                   delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                   det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0, image_slice_offset=0.0,
                   sigma_y=None, snr_db=40.0, sigma_x=None, sigma_p=None, p=1.2, q=2.0, T=1.0, num_neighbors=6,
                   sharpness=0.0, positivity=True, stop_threshold=0.02, max_iterations=100,
                   NHICD=False, num_threads=None, verbose=1, lib_path=__lib_path):

    sino_downsampled = half_res_sino(sino)

    num_image_rows_downsampled = int(np.ceil(num_image_rows/2))
    num_image_cols_downsampled = int(np.ceil(num_image_cols/2))
    num_image_slices_downsampled = int(np.ceil(num_image_slices/2))

    delta_det_channel_downsampled = delta_det_channel * 2
    delta_det_row_downsampled = delta_det_row * 2

    delta_pixel_image_downsampled = delta_pixel_image * 2

    x = cone3D.recon(sino_downsampled, angles, dist_source_detector, magnification,
                     weights=weights, weight_type=weight_type, init_image=init_image,
                     prox_image=prox_image,
                     num_image_rows=num_image_rows_downsampled, num_image_cols=num_image_cols_downsampled,
                     num_image_slices=num_image_slices_downsampled,
                     delta_det_channel=delta_det_channel_downsampled, delta_det_row=delta_det_row_downsampled,
                     delta_pixel_image=delta_pixel_image_downsampled,
                     det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                     rotation_offset=rotation_offset, image_slice_offset=image_slice_offset,
                     sigma_y=sigma_y, snr_db=snr_db, sigma_x=sigma_x, sigma_p=sigma_p, p=p, q=q, T=T,
                     num_neighbors=num_neighbors,
                     sharpness=sharpness, positivity=positivity, max_resolutions=0, stop_threshold=stop_threshold,
                     max_iterations=max_iterations,
                     NHICD=NHICD, num_threads=num_threads, verbose=verbose, lib_path=lib_path)

    return x
