import numpy as np
import os
import mbircone.cone3D as cone3D

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


def half_res_recon(sino, angles, dist_source_detector, magnification,
                   weights=None, weight_type='unweighted', init_image=0.0, prox_image=None,
                   num_image_rows=None, num_image_cols=None, num_image_slices=None,
                   delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                   det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0, image_slice_offset=0.0,
                   sigma_y=None, snr_db=40.0, sigma_x=None, sigma_p=None, p=1.2, q=2.0, T=1.0, num_neighbors=6,
                   sharpness=0.0, positivity=True, stop_threshold=0.02, max_iterations=100,
                   NHICD=False, num_threads=None, verbose=1, lib_path=__lib_path):
    """ Compute 3D cone beam MBIR reconstruction at half-resolution, then upsample to match the original image size.

    Args:
        sino (float, ndarray): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
        angles (float, ndarray): 1D array of view angles in radians.
        dist_source_detector (float): Distance between the X-ray source and the detector in units of :math:`ALU`.
        magnification (float): Magnification of the cone-beam geometry defined as
            (source to detector distance)/(source to center-of-rotation distance).

        weights (float, ndarray, optional): [Default=None] 3D weights array with same shape as ``sino``.
            If ``weights`` is not supplied, then ``cone3D.calc_weights`` is used to set weights using ``weight_type``.
        weight_type (string, optional): [Default='unweighted'] Type of noise model used for data.

                - ``'unweighted'`` corresponds to unweighted reconstruction;
                - ``'transmission'`` is the correct weighting for transmission CT with constant dosage;
                - ``'transmission_root'`` is commonly used with transmission CT data to improve image homogeneity;
                - ``'emission'`` is appropriate for emission CT data.
        init_image (float, ndarray, optional): [Default=0.0] Initial value of reconstruction image, specified by either
            a scalar value or a 3D numpy array with shape (num_img_slices, num_img_rows, num_img_cols).
        prox_image (float, ndarray, optional): [Default=None] 3D proximal map input image with shape
            (num_img_slices, num_img_rows, num_img_cols).

        num_image_rows (int, optional): [Default=None] Number of rows in reconstructed image.
            If None, automatically set by ``cone3D.auto_image_size``.
        num_image_cols (int, optional): [Default=None] Number of columns in reconstructed image.
            If None, automatically set by ``cone3D.auto_image_size``.
        num_image_slices (int, optional): [Default=None] Number of slices in reconstructed image.
            If None, automatically set by ``cone3D.auto_image_size``.

        delta_det_channel (float, optional): [Default=1.0] Detector channel spacing in :math:`ALU`.
        delta_det_row (float, optional): [Default=1.0] Detector row spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Image pixel spacing in :math:`ALU`.
            If None, automatically set to ``delta_pixel_detector/magnification``.

        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a row.
        det_row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector
            to the source-detector line along a column.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line
            to axis of rotation in the object space.
            This is normally set to zero.
        image_slice_offset (float, optional): [Default=0.0] Vertical offset of the image in units of :math:`ALU`.

        sigma_y (float, optional): [Default=None] Forward model regularization parameter.
            If None, automatically set with ``cone3D.auto_sigma_y``.
        snr_db (float, optional): [Default=40.0] Assumed signal-to-noise ratio of the data in :math:`dB`.
            Ignored if ``sigma_y`` is not None.
        sigma_x (float, optional): [Default=None] qGGMRF prior model regularization parameter.
            If None, automatically set with ``cone3D.auto_sigma_x`` as a function of ``sharpness``.
            If ``prox_image`` is given, ``sigma_p`` is used instead of ``sigma_x`` in the reconstruction.
        sigma_p (float, optional): [Default=None] Proximal map regularization parameter.
            If None, automatically set with ``cone3D.auto_sigma_p`` as a function of ``sharpness``.
            Ignored if ``prox_image`` is None.
        p (float, optional): [Default=1.2] Scalar value in range :math:`[1,2]` that specifies qGGMRF shape parameter.
        q (float, optional): [Default=2.0] Scalar value in range :math:`[p,1]` that specifies qGGMRF shape parameter.
        T (float, optional): [Default=1.0] Scalar value :math:`>0` that specifies the qGGMRF threshold parameter.
        num_neighbors (int, optional): [Default=6] Possible values are :math:`{26,18,6}`.
            Number of neighbors in the qGGMRF neighborhood. More neighbors results in a better
            regularization but a slower reconstruction.

        sharpness (float, optional): [Default=0.0] Sharpness of reconstruction.
            ``sharpness=0.0`` is neutral; ``sharpness>0`` increases sharpness; ``sharpness<0`` reduces sharpness.
            Used to calculate ``sigma_x`` and ``sigma_p``.
            Ignored if ``sigma_x`` is not None in qGGMRF mode, or if ``sigma_p`` is not None in proximal map mode.
        positivity (bool, optional): [Default=True] Determines if positivity constraint will be enforced.
        max_resolutions (int, optional): [Default=None] Integer :math:`\geq 0` that specifies the maximum number of grid
            resolutions used to solve MBIR reconstruction problem.
            If None, automatically set by ``cone3D.auto_max_resolutions``.
        stop_threshold (float, optional): [Default=0.02] Relative update stopping threshold, in percent, where relative
            update is given by (average value change) / (average voxel value).
        max_iterations (int, optional): [Default=100] Maximum number of iterations before stopping.

        NHICD (bool, optional): [Default=False] If True, uses non-homogeneous ICD updates.
        num_threads (int, optional): [Default=None] Number of compute threads requested when executed.
            If None, this is set to the number of cores in the system.
        verbose (int, optional): [Default=1] Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal
            reconstruction progress information, and 2 prints the full information.
        lib_path (str, optional): [Default=~/.cache/mbircone] Path to directory containing library of
            forward projection matrices.

    Returns:
        (float, ndarray): 3D reconstruction image with shape (num_img_slices, num_img_rows, num_img_cols) in units of
        :math:`ALU^{-1}`.
    """

    # Downsample by a factor of 2
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

    # upsample by a factor of 2
    x = x.repeat(2,0).repeat(2,1).repeat(2,2)

    return x
