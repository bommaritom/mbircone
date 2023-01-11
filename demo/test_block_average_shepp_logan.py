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


"""
The goal of this script is to test block average (say more)
"""
print('\t The goal of this script is to test block average (say more)\
\n\t (say more) \n')


# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

# Change the parameters below for your own use case.

# Detector size
num_det_rows = 128
num_det_channels = 128
# Geometry parameters
magnification = 1.0  # Ratio of (source to detector)/(source to center of rotation)
dist_factor = 1.0
dist_source_detector = dist_factor * 10 * num_det_channels  # distance from source to detector in ALU
# number of projection views
num_views = 64
# projection angles will be uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
# qGGMRF recon parameters
sharpness = 0.0  # Controls regularization: larger => sharper; smaller => smoother
T = 0.1  # Controls edginess of reconstruction
# display parameters
vmin = 0.10
vmax = 0.12

# Size of phantom
num_phantom_rows = 128
num_phantom_cols = 128
num_phantom_slices = 128
scale = 1.0
delta_pixel_phantom = 1

# Calculate scaling factor for Shepp Logan phantom so that projections are physically realistic -log attenuation values
SL_phantom_density_scale = 4.0*magnification/(scale*num_phantom_rows)



# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/test_block_average/'
os.makedirs(save_path, exist_ok=True)

print('Genrating 3D Shepp Logan phantom ...\n')
######################################################################################
# Generate a 3D shepp logan phantom
######################################################################################
phantom = mbircone.phantom.gen_shepp_logan_3d(num_phantom_rows, num_phantom_cols, num_phantom_slices)
phantom = SL_phantom_density_scale*phantom
print('Phantom shape = ', np.shape(phantom))

######################################################################################
# Generate synthetic sinograms
######################################################################################

print('Generating synthetic sinograms ...')

sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification,
                               delta_pixel_image=delta_pixel_phantom)

sino_block_average = np.zeros((num_views, (num_det_rows+1)//2, (num_det_channels+1)//2))

for view in range(num_views):
    sino_block_average[view] = block_average(sino[view], 2, 2)

print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)


######################################################################################
# Perform 3D MBIR reconstruction using qGGMRF prior
######################################################################################

print('Performing 3D qGGMRF reconstruction ...\n')
# recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
#                               sharpness=sharpness, T=T, max_resolutions=0)

recon_low_res = mbircone.cone3D.recon(sino_block_average, angles, dist_source_detector, magnification,
                                      det_channel_offset=0, det_row_offset=0,
                                      sharpness=sharpness, T=T, max_resolutions=0)

######################################################################################
# Generate phantom, synthetic sinogram, and reconstruction images
######################################################################################
# # sinogram images
for view_idx in [0, num_views // 4, num_views // 2]:
    view_angle = int(angles[view_idx] * 180 / np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle}',
               filename=os.path.join(save_path, f'sino-view-angle-{view_angle}.png'))
#
for view_idx in [0, num_views // 4, num_views // 2]:
    view_angle = int(angles[view_idx] * 180 / np.pi)
    plot_image(sino_block_average[view_idx, :, :], title=f'sinogram block average view angle {view_angle}',
               filename=os.path.join(save_path, f'sino-block-average-view-angle-{view_angle}.png'))


# # Print phantom images
# display_slice_phantom = num_slices_phantom // 2
# display_x_phantom = num_rows_phantom // 2
# display_y_phantom = num_cols_phantom // 2
#
#
# # phantom images
# plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
#            filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
# plot_image(phantom[:, display_x_phantom, :], title=f'phantom, coronal slice {display_x_phantom}',
#            filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
# plot_image(phantom[:, :, display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
#            filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)


# # Print recon images
# num_slices_recon, num_rows_recon, num_cols_recon = np.shape(recon)
#
# display_slice_recon = num_slices_recon // 2
# display_x_recon = num_rows_recon // 2
# display_y_recon = num_cols_recon // 2
#
# # recon images
# plot_image(recon[display_slice_recon], title=f'recon, axial slice {display_slice_recon}',
#            filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
# plot_image(recon[:, display_x_recon, :], title=f'recon, coronal slice {display_x_recon}',
#            filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
# plot_image(recon[:, :, display_y_recon], title=f'recon, sagittal slice {display_y_recon}',
#            filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)

# Print recon low res images
num_slices_recon_low_res, num_rows_recon_low_res, num_cols_recon_low_res = np.shape(recon_low_res)

display_slice_recon_low_res = num_slices_recon_low_res // 2
display_x_recon_low_res = num_rows_recon_low_res // 2
display_y_recon_low_res = num_cols_recon_low_res // 2

# recon low res images
plot_image(recon_low_res[display_slice_recon_low_res], title=f'recon low res, axial slice {display_slice_recon_low_res}',
           filename=os.path.join(save_path, 'recon_low_res_axial.png'), vmin=vmin, vmax=.05)
plot_image(recon_low_res[:, display_x_recon_low_res, :], title=f'recon low res, coronal slice {display_x_recon_low_res}',
           filename=os.path.join(save_path, 'recon_low_res_coronal.png'), vmin=vmin, vmax=.05)
plot_image(recon_low_res[:, :, display_y_recon_low_res], title=f'recon low res, sagittal slice {display_y_recon_low_res}',
           filename=os.path.join(save_path, 'recon_low_res_sagittal.png'), vmin=vmin, vmax=.05)

print(f"Images saved to {save_path}.")
input("Press Enter")
