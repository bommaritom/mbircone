import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif


"""
This script is a demonstration of the laminography reconstruction algorithm. Demo functionality includes
 * Generating a synthetic phantom with a single lit pixel.
 * Depending on the value of test_number:
   * A) Generating a synthetic sinogram with delta_det_channel=2, delta_det_row=2.
   * B) Generating a synthetic sinogram with delta_pixel_image=2.
   * C) Generating a synthetic sinogram with image_slice_offset=1.
   * D) Generating a synthetic sinogram with det_channel_offset=1.
 * Displaying the results.
"""

print('This script is a demonstration of the laminography reconstruction algorithm. Demo functionality includes \
\n\t * Generating a synthetic phantom with a single lit pixel. \
\n\t * Depending on the value of test_number: \
\n\t\t A) Generating a synthetic sinogram with delta_det_channel=2, delta_det_row=2. \
\n\t\t B) Generating a synthetic sinogram with delta_pixel_image=2. \
\n\t\t C) Generating a synthetic sinogram with image_slice_offset=1. \
\n\t\t D) Generating a synthetic sinogram with det_channel_offset=1. \
\n\t * Displaying the results.')

test_code = 'Default'

# Laminographic angle
theta_degrees = 60
# Convert to radians
theta_radians = theta_degrees * (np.pi/180)

# Detector size
num_det_rows = 9
num_det_channels = 9
# number of projection views
num_views = 64
# projection angles will be uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# Size of phantom
num_slices_phantom = 9
num_rows_phantom = 9
num_cols_phantom = 9

# Compute projection angles uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/test_project_lamino/'
os.makedirs(save_path, exist_ok=True)

phantom = np.zeros((num_slices_phantom, num_rows_phantom, num_cols_phantom))
phantom[4, 4, 4] = 1

######################################################################################
# Generate synthetic sinogram
######################################################################################

print('Generating synthetic sinogram ...')

if test_code == 'A':
    # Generate a synthetic sinogram with delta_det_channel=2.
    sino = mbircone.laminography.project_lamino(phantom, angles, theta_radians,
                                                num_det_rows, num_det_channels,
                                                delta_det_channel=2, delta_det_row=2)
elif test_code == 'B':
    # Generate a synthetic sinogram with delta_pixel_image=2
    sino = mbircone.laminography.project_lamino(phantom, angles, theta_radians,
                                                num_det_rows, num_det_channels,
                                                delta_pixel_image=2)
elif test_code == 'C':
    # Generate a synthetic sinogram with image_slice_offset=1
    sino = mbircone.laminography.project_lamino(phantom, angles, theta_radians,
                                                num_det_rows, num_det_channels,
                                                image_slice_offset=1)
elif test_code == 'D':
    # Generate a synthetic sinogram with det_channel_offset=1
    sino = mbircone.laminography.project_lamino(phantom, angles, theta_radians,
                                                num_det_rows, num_det_channels,
                                                det_channel_offset=1)
else:
    # Generate a synthetic sinogram normally.
    sino = mbircone.laminography.project_lamino(phantom, angles, theta_radians,
                                                num_det_rows, num_det_channels)

#####################################################################################
# Generate phantom, synthetic sinogram, and reconstruction images
#####################################################################################
# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
        view_angle = int(angles[view_idx]*180/np.pi)
        plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ',
                   filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))

display_phantom = phantom

# Set display indexes for phantom and recon images
display_slice_image = display_phantom.shape[0] // 2
display_x_image = display_phantom.shape[1] // 2
display_y_image = display_phantom.shape[2] // 2

# phantom images
plot_image(display_phantom[display_slice_image], title=f'phantom, axial slice {display_slice_image}',
           filename=os.path.join(save_path, 'phantom_axial.png'))
plot_image(display_phantom[:,display_x_image,:], title=f'phantom, coronal slice {display_x_image}',
           filename=os.path.join(save_path, 'phantom_coronal.png'))
plot_image(display_phantom[:,:,display_y_image], title=f'phantom, sagittal slice {display_y_image}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'))

# print(sino[0,:,:])

print(f"Images saved to {save_path}.")
input("Press Enter")
