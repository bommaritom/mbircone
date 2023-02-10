import os
import numpy as np
import mbircone
from demo_utils import plot_image, plot_gif


"""
This script is a demonstration of the laminography reconstruction algorithm. Demo functionality includes
 * Generating a synthetic phantom with a single lit pixel.
 * Generating a synthetic sinogram by running project_lamino.
 * Depending on the value of test_number:
   * A) Performing a 3D qGGMRF reconstruction with delta_det_channel=2, delta_det_row=2.
   * B) Performing a 3D qGGMRF reconstruction with delta_pixel_image=2.
   * C) Performing a 3D qGGMRF reconstruction with image_slice_offset=1.
   * D) Performing a 3D qGGMRF reconstruction with det_channel_offset=1.
   * Displaying the results.
"""
print('This script is a demonstration of the laminography reconstruction algorithm. Demo functionality includes \
\n\t * Generating a synthetic phantom with a single lit pixel. \
\n\t * Generating a synthetic sinogram by running project_lamino. \
\n\t * Depending on the value of test_number: \
\n\t\t A) Performing a 3D qGGMRF reconstruction with delta_det_channel=2. \
\n\t\t B) Performing a 3D qGGMRF reconstruction with delta_pixel_image=2. \
\n\t\t C) Performing a 3D qGGMRF reconstruction with image_slice_offset=1. \
\n\t\t D) Performing a 3D qGGMRF reconstruction with det_channel_offset=1. \
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

# Set reconstruction parameters
sharpness = 0.0                             # Controls regularization: larger => sharper; smaller => smoother
T = 0.1                                     # Controls edginess of reconstruction
snr_db = 30

# Size of phantom
num_slices_phantom = 9
num_rows_phantom = 9
num_cols_phantom = 9

num_image_slices = 9
num_image_cols = 9
num_image_rows = 9

# Compute projection angles uniformly spaced within the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/test_recon_lamino/'
os.makedirs(save_path, exist_ok=True)

phantom = np.zeros((num_slices_phantom, num_rows_phantom, num_cols_phantom))
phantom[4, 4, 4] = 1

######################################################################################
# Generate synthetic sinogram
######################################################################################

print('Generating synthetic sinogram ...')
sino = mbircone.laminography.project_lamino(phantom, angles, theta_radians,
                                            num_det_rows, num_det_channels)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)


######################################################################################
# Perform 3D qGGMRF reconstruction
######################################################################################

print('Performing 3D qGGMRF reconstruction ...')

if test_code == 'A':
    # Perform a 3D qGGMRF reconstruction with delta_det_channel=2.
    recon = mbircone.laminography.recon_lamino(sino, angles, theta_radians,
                                               num_image_rows=num_image_rows,
                                               num_image_cols=num_image_cols, num_image_slices=num_image_slices,
                                               delta_det_channel=2, delta_det_row=2,
                                               sharpness=sharpness, snr_db=snr_db)
elif test_code == 'B':
    # Perform a 3D qGGMRF reconstruction with delta_pixel_image=2.
    recon = mbircone.laminography.recon_lamino(sino, angles, theta_radians,
                                               num_image_rows=num_image_rows,
                                               num_image_cols=num_image_cols, num_image_slices=num_image_slices,
                                               delta_pixel_image=2,
                                               sharpness=sharpness, snr_db=snr_db)
elif test_code == 'C':
    # Perform a 3D qGGMRF reconstruction with image_slice_offset=1.
    recon = mbircone.laminography.recon_lamino(sino, angles, theta_radians,
                                               num_image_rows=num_image_rows,
                                               num_image_cols=num_image_cols, num_image_slices=num_image_slices,
                                               image_slice_offset=1,
                                               sharpness=sharpness, snr_db=snr_db)
elif test_code == 'D':
    # Perform a 3D qGGMRF reconstruction with delta_channel_offset=1.
    recon = mbircone.laminography.recon_lamino(sino, angles, theta_radians,
                                               num_image_rows=num_image_rows,
                                               num_image_cols=num_image_cols, num_image_slices=num_image_slices,
                                               det_channel_offset=1,
                                               sharpness=sharpness, snr_db=snr_db)
else:
    # Run a regular reconstruction
    recon = mbircone.laminography.recon_lamino(sino, angles, theta_radians,
                                               num_image_rows=num_image_rows,
                                               num_image_cols=num_image_cols, num_image_slices=num_image_slices,
                                               sharpness=sharpness, snr_db=snr_db)
print('recon shape = ', np.shape(recon))


#####################################################################################
# Generate phantom, synthetic sinogram, and reconstruction images
#####################################################################################
# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
        view_angle = int(angles[view_idx]*180/np.pi)
        plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ',
                   filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))

display_phantom = phantom
display_recon = recon

# Set display indexes for phantom and recon images
display_slice_image = display_phantom.shape[0] // 2
display_x_image = display_phantom.shape[1] // 2
display_y_image = display_phantom.shape[2] // 2

display_slice_recon = display_recon.shape[0] // 2
display_x_recon = display_recon.shape[1] // 2
display_y_recon = display_recon.shape[2] // 2

# phantom images
plot_image(display_phantom[display_slice_image], title=f'phantom, axial slice {display_slice_image}',
           filename=os.path.join(save_path, 'phantom_axial.png'))
plot_image(display_phantom[:,display_x_image,:], title=f'phantom, coronal slice {display_x_image}',
           filename=os.path.join(save_path, 'phantom_coronal.png'))
plot_image(display_phantom[:,:,display_y_image], title=f'phantom, sagittal slice {display_y_image}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'))
# recon images
plot_image(display_recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}, '
                                                     f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_axial.png'))
plot_image(display_recon[:, display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}, '
                                                      f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_coronal.png'))
plot_image(display_recon[:, :, display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}, '
                                                       f'Θ='+str(theta_degrees)+' degrees',
           filename=os.path.join(save_path, 'recon_sagittal.png'))

print(f"Images saved to {save_path}.")
input("Press Enter")