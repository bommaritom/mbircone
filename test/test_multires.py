import os
import numpy as np
import mbircone
import matplotlib.pyplot as plt
from test_utils import block_average_sino, block_average_3D, plot_image
from scipy import signal as sgn


"""
This script is a demonstration of a single multires reconstruction.
Demo functionality includes:
 * Generating a 3D Shepp Logan phantom;
 * Forward projecting the Shepp Logan phantom to form a synthetic sinogram;
 * Computing a 3D reconstruction from the sinogram at half-resolution;
 * Upsampling the reconstructed image;
 * Displaying the results.
"""
print('This script is a demonstration of a single multires reconstruction.\
\nDemo functionality includes:\
\n\t * Generating a 3D Shepp Logan phantom; \
\n\t * Forward projecting the Shepp Logan phantom to form a synthetic sinogram;\
\n\t * Computing a 3D reconstruction from the sinogram at half-resolution;\
\n\t * Upsampling the reconstructed image;\
\n\t * Displaying the results.\n')


# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

# Change the parameters below for your own use case.

# Detector and geometry parameters
num_det_rows = 128                           # Number of detector rows
num_det_channels = 128                       # Number of detector channels
magnification = 1.0                          # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 3.0*num_det_channels  # Distance from source to detector in ALU
num_views = 128                              # Number of projection views

# Generate uniformly spaced view angles in the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# Set reconstruction parameters
sharpness = 0.0                             # Controls regularization: larger => sharper; smaller => smoother
T = 0.1                                     # Controls edginess of reconstruction
delta_pixel = 1.0/magnification             # Pixel pitch for default reconstruction resolution

# Set phantom generation parameters
num_phantom_slices = num_det_rows           # Set number of slides = to the number of detector rows
num_phantom_rows = num_det_channels         # Make number of rows and columns = to number of detector columns
num_phantom_cols = num_det_channels
scale=1.0                                   # Determines the size of the phantom within the volume

# Calculate scaling factor for Shepp Logan phantom so that projections are physically realistic -log attenuation values
SL_phantom_density_scale = 4.0*magnification/(scale*num_phantom_rows)

# Set reconstruction ROI to be only the region containing the phantom
num_image_slices = int(scale*num_phantom_slices)
num_image_rows = int(scale*num_phantom_rows)
num_image_cols = int(scale*num_phantom_cols)

# Set display parameters for Shepp Logan phantom
vmin = SL_phantom_density_scale*1.0
vmax = SL_phantom_density_scale*1.2

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/test_multires/'
os.makedirs(save_path, exist_ok=True)

######################################################################################
# Generate phantom_A
######################################################################################
phantom_A = mbircone.phantom.gen_shepp_logan_3d(num_phantom_rows, num_phantom_cols,
                                                num_phantom_slices, scale=scale)
phantom_A = SL_phantom_density_scale*phantom_A


######################################################################################
# Generate sino_A
######################################################################################
sino_A = mbircone.cone3D.project(phantom_A, angles,
                                 num_det_rows=128, num_det_channels=128,
                                 dist_source_detector=dist_source_detector,
                                 magnification=magnification,
                                 delta_det_channel=1.0, delta_det_row=1.0,
                                 delta_pixel_image=1.0)


######################################################################################
# Generate sino_A_block_average by taking the view-wise block average of sino_A
######################################################################################
sino_A_block_average = block_average_sino(sino_A)


######################################################################################
# Generate recon_A by performing qGGMRF reconstruction on sino_A_block_average
######################################################################################
recon_A = mbircone.cone3D.recon(sino_A_block_average, angles,
                                dist_source_detector=dist_source_detector,
                                magnification=magnification,
                                delta_det_channel=2.0, delta_det_row=2.0,
                                delta_pixel_image=2.0,
                                max_resolutions=0)


######################################################################################
# Generate phantom_B by performing 3D block-averaging on phantom_A
######################################################################################
phantom_B = block_average_3D(phantom_A, 2, 2, 2)


######################################################################################
# Generate sino_B
######################################################################################
sino_B = mbircone.cone3D.project(phantom_B, angles,
                                 num_det_rows=64, num_det_channels=64,
                                 dist_source_detector=dist_source_detector,
                                 magnification=magnification,
                                 delta_det_channel=2.0, delta_det_row=2.0,
                                 delta_pixel_image=2.0)

######################################################################################
# Generate recon_B by performing qGGMRF reconstruction on sino_B
######################################################################################
recon_B = mbircone.cone3D.recon(sino_B, angles,
                                dist_source_detector=dist_source_detector,
                                magnification=magnification,
                                delta_det_channel=2.0, delta_det_row=2.0,
                                delta_pixel_image=2.0,
                                max_resolutions=0)


######################################################################################
# Sinogram error image
######################################################################################
sino_diff = np.abs(sino_A_block_average - sino_B)


######################################################################################
# Display results
######################################################################################

# Suppress max figure warning
plt.rcParams.update({'figure.max_open_warning': 0})

# phantom_A
plot_image(phantom_A[64], title=f'phantom_A, axial slice {64}',
           filename=os.path.join(save_path, 'phantom_A_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_A[:,64,:], title=f'phantom_A, coronal slice {64}',
           filename=os.path.join(save_path, 'phantom_A_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_A[:,:,64], title=f'phantom_A, sagittal slice {64}',
           filename=os.path.join(save_path, 'phantom_A_sagittal.png'), vmin=vmin, vmax=vmax)

# sino_A
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino_A[view_idx, :, :], title=f'sino_A view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-A-shepp-logan-3D-view_angle{view_angle}.png'))

# sino_A_block_average
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino_A_block_average[view_idx, :, :], title=f'sino_A_block_average view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-A-block-average-shepp-logan-3D-view_angle{view_angle}.png'))

# recon_A
plot_image(recon_A[32], title=f'recon_A, axial slice {32}',
           filename=os.path.join(save_path, 'recon_A_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon_A[:,32,:], title=f'recon_A, coronal slice {32}',
           filename=os.path.join(save_path, 'recon_A_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon_A[:,:,32], title=f'recon_A, sagittal slice {32}',
           filename=os.path.join(save_path, 'recon_A_sagittal.png'), vmin=vmin, vmax=vmax)

print(f"Images saved to {save_path}.")
input("Press Enter")

# phantom_B
plot_image(phantom_B[32], title=f'phantom_B, axial slice {32}',
           filename=os.path.join(save_path, 'phantom_B_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_B[:,32,:], title=f'phantom_B, coronal slice {32}',
           filename=os.path.join(save_path, 'phantom_B_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom_B[:,:,32], title=f'phantom_B, sagittal slice {32}',
           filename=os.path.join(save_path, 'phantom_B_sagittal.png'), vmin=vmin, vmax=vmax)

# sino_B
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino_B[view_idx, :, :], title=f'sino_B view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-B-shepp-logan-3D-view_angle{view_angle}.png'))

# recon_B
plot_image(recon_B[32], title=f'recon_B, axial slice {32}',
           filename=os.path.join(save_path, 'recon_B_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon_B[:,32,:], title=f'recon_B, coronal slice {32}',
           filename=os.path.join(save_path, 'recon_B_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon_B[:,:,32], title=f'recon_B, sagittal slice {32}',
           filename=os.path.join(save_path, 'recon_B_sagittal.png'), vmin=vmin, vmax=vmax)

# sino_diff
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino_diff[view_idx, :, :], title=f'sino_diff view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-diff-shepp-logan-3D-view_angle{view_angle}.png'))


print(f"Images saved to {save_path}.") 
input("Press Enter")

