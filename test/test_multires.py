import os
import numpy as np
import mbircone
import matplotlib.pyplot as plt
from test_utils import block_average_sino, block_average_3D, plot_image, apply_gaussian_filter_to_sino, zoom_sino
from skimage.transform import resize
from scipy import signal as sgn


"""
The purpose of this script is to perform a qGGMRF reconstruction after
filtering the sinogram with a Gaussian filter, and downsampling.
Functionality includes:
 * Generating a 3D Shepp Logan phantom;
 * Forward projecting the Shepp Logan phantom to form a synthetic sinogram;
 * Filtering the sinogram with a Gaussian filter;
 * Computing a 3D reconstruction from the sinogram at half-resolution;
 * Upsampling the reconstructed image;
 * Displaying the results.
"""
print('The purpose of this script is to perform a qGGMRF reconstruction after\
\nfiltering the sinogram with a Gaussian filter, and downsampling.\
\nFunctionality includes:\
\n\t * Generating a 3D Shepp Logan phantom; \
\n\t * Forward projecting the Shepp Logan phantom to form a synthetic sinogram;\
\n\t * Filtering the sinogram with a Gaussian filter;\
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
save_path = f'output/init_image_comparison/'
os.makedirs(save_path, exist_ok=True)

######################################################################################
# Generate phantom
######################################################################################
phantom = mbircone.phantom.gen_shepp_logan_3d(num_phantom_rows, num_phantom_cols,
                                                num_phantom_slices, scale=scale)
phantom = SL_phantom_density_scale*phantom


######################################################################################
# Generate sino
######################################################################################
sino = mbircone.cone3D.project(phantom, angles,
                                 num_det_rows=128, num_det_channels=128,
                                 dist_source_detector=dist_source_detector,
                                 magnification=magnification,
                                 delta_det_channel=1.0, delta_det_row=1.0,
                                 delta_pixel_image=1.0)

######################################################################################
# Generate init_image_A by performing qGGMRF with sino
######################################################################################
init_image_A = mbircone.cone3D.recon(sino, angles,
                                     dist_source_detector=dist_source_detector,
                                     magnification=magnification,
                                     delta_det_channel=1.0, delta_det_row=1.0,
                                     delta_pixel_image=2.0,
                                     max_resolutions=0)
init_image_A = np.repeat(init_image_A, 2, axis=0)
init_image_A = np.repeat(init_image_A, 2, axis=1)
init_image_A = np.repeat(init_image_A, 2, axis=2)


####################################################c##################################
# Generate low_res_sino_B by performing block-averaging and Gaussian filtering
######################################################################################
sino_gaussian_filter = apply_gaussian_filter_to_sino(sino, sigma=1)
low_res_sino_B = block_average_sino(sino_gaussian_filter)
# low_res_sino_B = resize(sino, (num_views, num_det_rows // 2, num_det_channels // 2), anti_aliasing=True)
# low_res_sino_B = zoom_sino(sino)

######################################################################################
# Generate init_image_B by performing qGGMRF with low_res_sino_B
######################################################################################
init_image_B = mbircone.cone3D.recon(low_res_sino_B, angles,
                                     dist_source_detector=dist_source_detector,
                                     magnification=magnification,
                                     delta_det_channel=2.0, delta_det_row=2.0,
                                     delta_pixel_image=2.0,
                                     max_resolutions=0)
init_image_B = np.repeat(init_image_B, 2, axis=0)
init_image_B = np.repeat(init_image_B, 2, axis=1)
init_image_B = np.repeat(init_image_B, 2, axis=2)

######################################################################################
# Generate recon_A and recon_B by performing qGGMRF reconstruction with init_image_A
# and init_image_B
######################################################################################
print("Generating recon A.")
recon_A = mbircone.cone3D.recon(sino, angles,
                                init_image=init_image_A,
                                dist_source_detector=dist_source_detector,
                                magnification=magnification,
                                delta_det_channel=1.0, delta_det_row=1.0,
                                delta_pixel_image=1.0,
                                max_resolutions=0)

print("Generating recon B.")
recon_B = mbircone.cone3D.recon(sino, angles,
                                init_image=init_image_B,
                                dist_source_detector=dist_source_detector,
                                magnification=magnification,
                                delta_det_channel=1.0, delta_det_row=1.0,
                                delta_pixel_image=1.0,
                                max_resolutions=0)

######################################################################################
# Display results
######################################################################################

view_slice = num_phantom_slices // 2
view_row = num_phantom_rows // 2
view_col = num_phantom_cols // 2

# phantom
plot_image(phantom[view_slice], title=f'phantom, axial slice {view_slice}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,view_row,:], title=f'phantom, coronal slice {view_row}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,view_col], title=f'phantom, sagittal slice {view_row}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)

# sino
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino[view_idx, :, :], title=f'sino view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))

# low res sino
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(low_res_sino_B[view_idx, :, :], title=f'low res sino B filtered view angle {view_angle} ',
               filename=os.path.join(save_path, f'low-res-sino-B-shepp-logan-3D-view_angle{view_angle}.png'))

# init_image_A
plot_image(init_image_A[view_slice], title=f'init image A, axial slice {view_slice}',
           filename=os.path.join(save_path, 'init_image_A_axial.png'), vmin=vmin, vmax=vmax)
plot_image(init_image_A[:,view_row,:], title=f'init image A, coronal slice {view_row}',
           filename=os.path.join(save_path, 'init_image_A_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(init_image_A[:,:,view_col], title=f'init image A, sagittal slice {view_col}',
           filename=os.path.join(save_path, 'init_image_A_sagittal.png'), vmin=vmin, vmax=vmax)

# init_image_B
plot_image(init_image_B[view_slice], title=f'init image B, axial slice {view_slice}',
           filename=os.path.join(save_path, 'init_image_B_axial.png'), vmin=vmin, vmax=vmax)
plot_image(init_image_B[:,view_row,:], title=f'init image B, coronal slice {view_row}',
           filename=os.path.join(save_path, 'init_image_B_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(init_image_B[:,:,view_col], title=f'init image B, sagittal slice {view_col}',
           filename=os.path.join(save_path, 'init_image_B_sagittal.png'), vmin=vmin, vmax=vmax)

# recon_A
plot_image(recon_A[view_slice], title=f'recon A, axial slice {view_slice}',
           filename=os.path.join(save_path, 'recon_A_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon_A[:,view_row,:], title=f'recon A, coronal slice {view_row}',
           filename=os.path.join(save_path, 'recon_A_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon_A[:,:,view_col], title=f'recon A, sagittal slice {view_col}',
           filename=os.path.join(save_path, 'recon_A_sagittal.png'), vmin=vmin, vmax=vmax)

# recon_B
plot_image(recon_B[view_slice], title=f'recon B, axial slice {view_slice}',
           filename=os.path.join(save_path, 'recon_B_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon_B[:,view_row,:], title=f'recon B, coronal slice {view_row}',
           filename=os.path.join(save_path, 'recon_B_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon_B[:,:,view_col], title=f'recon B, sagittal slice {view_col}',
           filename=os.path.join(save_path, 'recon_B_sagittal.png'), vmin=vmin, vmax=vmax)



print(f"Images saved to {save_path}.") 
input("Press Enter")

