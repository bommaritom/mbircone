import os
import numpy as np
from matplotlib import pyplot as plt
import mbircone
from demo_utils import plot_image
from scipy import signal as sgn


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

    # get average of the 2x2 blocks
    image_block_avg = np.mean(image_reshape, axis=2)
    image_block_avg = np.mean(image_block_avg, axis=2)

    return image_block_avg


def get_kernel(sampling_rate, order):
    """
    Return a sampled 2D sinc filter of size 2*order+1, corresponding to the
    given sampling rate sampling_rate, with DC shift 1, windowed with hamming window
    """
    output = [np.sinc(x / sampling_rate)/sampling_rate for x in range(-order, order+1)]
    output *= np.hamming(2*order + 1)
    return np.outer(output, output)


"""
The goal of this script is to test a rudimentary block average function. Functionality includes
 * Forward-projecting the block-averaged phantom to generate a synthetic sinogram;
 * Generating a dummy 13x13x13 phantom with three lit pixels;
 * Performing pre-filtering and block-averaging;
 * Computing a 3D reconstruction from the sinogram using a qGGMRF prior model;
 * Displaying the results. 
"""
print('\t The goal of this script is to test a rudimentary block average function. Functionality includes\
\n\t * Forward-projecting the block-averaged phantom to generate a synthetic sinogram; \
\n\t * Generating a dummy 13x13x13 phantom with three lit pixels;\
\n\t * Performing pre-filtering and block-averaging;\
\n\t * Computing a 3D reconstruction from the sinogram using a qGGMRF prior model;\
\n\t * Displaying the results. \n')


# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

# Change the parameters below for your own use case.

# Detector size
num_det_rows = 13
num_det_channels = 13
# Geometry parameters
magnification = 1.0  # Ratio of (source to detector)/(source to center of rotation)
dist_factor = 1.0
dist_source_detector = dist_factor * 10 * num_det_channels  # distance from source to detector in ALU
# number of projection views
num_views = 64
# projection angles will be uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
# qGGMRF recon parameters
sharpness = 0.1  # Controls regularization: larger => sharper; smaller => smoother
T = 0.1  # Controls edginess of reconstruction
# display parameters
vmin = 0.10
vmax = 0.12

# Size of phantom
num_slices_phantom = 13
num_rows_phantom = 13
num_cols_phantom = 13
delta_pixel_phantom = 1

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/test_block_average/'
os.makedirs(save_path, exist_ok=True)

print('Genrating phantom ...')
######################################################################################
# Generate phantom
######################################################################################

phantom = np.zeros((num_slices_phantom, num_rows_phantom, num_cols_phantom))
phantom[num_slices_phantom//2, num_rows_phantom//2, num_cols_phantom//2] = 2
phantom[num_slices_phantom//2+2, num_rows_phantom//2+3, num_cols_phantom//2] = 1
phantom[num_slices_phantom//2, num_rows_phantom//2+2, num_cols_phantom//2] = 1

######################################################################################
# Generate synthetic sinograms
######################################################################################

print('Generating synthetic sinograms ...')

sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification,
                               delta_pixel_image=delta_pixel_phantom)

# prefilter to prepare for averaging + downsampling
kernel = get_kernel(2, 4)
sino_pre_filter = np.copy(sino)
for view_angle in range(np.shape(sino_pre_filter)[0]):
    sino_pre_filter[view_angle] = sgn.convolve(sino_pre_filter[view_angle], kernel, mode='same')

# perform block-averaging
sino_block_average = np.zeros((num_views, (num_det_rows+1)//2, (num_det_channels+1)//2))
for view in range(num_views):
    sino_block_average[view] = block_average(sino_pre_filter[view], 2, 2)


print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)


######################################################################################
# Perform 3D MBIR reconstruction using qGGMRF prior
######################################################################################

print('Performing 3D qGGMRF reconstruction ...\n')
recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                              sharpness=sharpness, T=T, max_resolutions=0)

recon_low_res = mbircone.cone3D.recon(sino_block_average, angles, dist_source_detector, magnification,
                                      det_channel_offset=-0.25, det_row_offset=-0.25,
                                      sharpness=sharpness, T=T, max_resolutions=0)

######################################################################################
# Generate phantom, synthetic sinogram, and reconstruction images
######################################################################################
# # sinogram images
for view_idx in [0, num_views // 4, num_views // 2]:
    view_angle = int(angles[view_idx] * 180 / np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle}',
               filename=os.path.join(save_path, f'sino-view-angle-{view_angle}.png'))

for view_idx in [0, num_views // 4, num_views // 2]:
    view_angle = int(angles[view_idx] * 180 / np.pi)
    plot_image(sino_pre_filter[view_idx, :, :], title=f'sinogram low pass view angle {view_angle}',
               filename=os.path.join(save_path, f'sino-low-pass-view-angle-{view_angle}.png'))


#
for view_idx in [0, num_views // 4, num_views // 2]:
    view_angle = int(angles[view_idx] * 180 / np.pi)
    plot_image(sino_block_average[view_idx, :, :], title=f'sinogram block average view angle {view_angle}',
               filename=os.path.join(save_path, f'sino-block-average-view-angle-{view_angle}.png'))


# Print phantom images
display_slice_phantom = num_slices_phantom // 2
display_x_phantom = num_rows_phantom // 2
display_y_phantom = num_cols_phantom // 2


# phantom images
plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'))
plot_image(phantom[:, display_x_phantom, :], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'))
plot_image(phantom[:, :, display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'))


# Print recon images
num_slices_recon, num_rows_recon, num_cols_recon = np.shape(recon)

display_slice_recon = num_slices_recon // 2
display_x_recon = num_rows_recon // 2
display_y_recon = num_cols_recon // 2

# recon images
plot_image(recon[display_slice_recon], title=f'recon, axial slice {display_slice_recon}',
           filename=os.path.join(save_path, 'recon_axial.png'))
plot_image(recon[:, display_x_recon, :], title=f'recon, coronal slice {display_x_recon}',
           filename=os.path.join(save_path, 'recon_coronal.png'))
plot_image(recon[:, :, display_y_recon], title=f'recon, sagittal slice {display_y_recon}',
           filename=os.path.join(save_path, 'recon_sagittal.png'))

# Print recon low res images
num_slices_recon_low_res, num_rows_recon_low_res, num_cols_recon_low_res = np.shape(recon_low_res)

display_slice_recon_low_res = num_slices_recon_low_res // 2
display_x_recon_low_res = num_rows_recon_low_res // 2
display_y_recon_low_res = num_cols_recon_low_res // 2

# recon low res images
plot_image(recon_low_res[display_slice_recon_low_res], title=f'recon low res, axial slice {display_slice_recon_low_res}',
           filename=os.path.join(save_path, 'recon_low_res_axial.png'))
plot_image(recon_low_res[:, display_x_recon_low_res, :], title=f'recon low res, coronal slice {display_x_recon_low_res}',
           filename=os.path.join(save_path, 'recon_low_res_coronal.png'))
plot_image(recon_low_res[:, :, display_y_recon_low_res], title=f'recon low res, sagittal slice {display_y_recon_low_res}',
           filename=os.path.join(save_path, 'recon_low_res_sagittal.png'))

print("Final reconstruction pixel values:" + str(recon_low_res[:, :, display_y_recon_low_res].round(3)))


# Print out "expected" reconstruction image; if the original phantom were block-averaged.

expected = np.zeros((7,7))
expected[3,3] = 0.5
expected[3,4] = 0.25
expected[4,4] = 0.125
expected[4,5] = 0.125

plot_image(expected, title=f'expected reconstruction, sagittal slice {display_slice_recon_low_res}',
           filename=os.path.join(save_path, 'expected_sagittal.png'))

print(f"Images saved to {save_path}.")
input("Press Enter")
