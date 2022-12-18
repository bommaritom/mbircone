import os
import numpy as np
from matplotlib import pyplot as plt
import mbircone
from demo_utils import plot_image
from scipy import signal as sgn

def get_kernel(sampling_rate, order):
    """
    Return a sampled 2D sinc filter of size 2*order+1, corresponding to the
    given sampling rate sampling_rate, with DC shift 1
    """
    output = [np.sinc(x / sampling_rate)/sampling_rate for x in range(-order, order+1)]
    output *= np.hamming(2*order + 1)
    return np.outer(output, output)

"""
The goal of this script is to determine whether we can reduce
the number of iterations in qGGMRF reconstruction by providing
an initial image. Use 'use_initial_image' to turn the initial
image generation on or off. Use 'filter_sinogram' to turn
sinogram convolutional filtering on or off.
"""
print('The goal of this script is to determine whether we can reduce\
\n\t the number of iterations in qGGMRF reconstruction by providing\
\n\t an initial image. Use \'use_initial_image\' to turn the initial\
\n\t image generation on or off. Use \'filter_sinogram\' to turn\
\n\t sinogram convolutional filtering on or off.\n')


# #################################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# #################################################################################

use_initial_image = True                     # If True, generate initial image using qGGMRF at half
                                             # pixel pitch.

filter_sinogram = True                       # If True, filter sinogram using the parameters
                                             # 'sampling_rate_kernel' and 'order_kernel'.

sampling_rate_kernel = 2.0                   # Sampling rate of 2D sinc function with Hamming window.
order_kernel = 4                             # Order of convolutional kernel. Support of kernel
                                             # is 2*(order_kernel)+1

# Detector and geometry parameters
num_det_rows = 128                           # Number of detector rows
num_det_channels = 128                       # Number of detector channels
magnification = 2.0                          # Ratio of (source to detector)/(source to center of rotation)
dist_source_detector = 3.0*num_det_channels  # Distance from source to detector in ALU
num_views = 64                               # Number of projection views

# Generate uniformly spaced view angles in the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

# Set reconstruction parameters
sharpness = 0.0                             # Controls regularization: larger => sharper; smaller => smoother
T = 0.1                                     # Controls edginess of reconstruction
delta_pixel = 1.0/magnification             # Pixel pitch for default reconstruction resolution

# Set phantom generation parameters
num_phantom_slices = num_det_rows           # Set number of slides = to the number of detector rows
num_phantom_rows = num_det_channels         # Make number of rows and columns = to number of
                                            # detector columns
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
save_path = f'output/test_filter_reconstruction/'
os.makedirs(save_path, exist_ok=True)

print('Genrating 3D Shepp Logan phantom ...\n')
######################################################################################
# Generate a 3D shepp logan phantom
######################################################################################
phantom = mbircone.phantom.gen_shepp_logan_3d(num_phantom_rows, num_phantom_cols, num_phantom_slices, scale=scale)
phantom = SL_phantom_density_scale*phantom
print('Phantom shape = ', np.shape(phantom))

######################################################################################
# Generate synthetic sinogram
######################################################################################
print('Generating synthetic sinogram ...\n')
sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification, delta_pixel_image=delta_pixel)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)

######################################################################################
# Generate initial image by running recon on lower pixel pitch
######################################################################################

if use_initial_image:

    # Generate convolutional filter to be applied to the sinogram.

    kernel = get_kernel(sampling_rate_kernel, order_kernel)

    if filter_sinogram:
        sino_filtered = np.copy(sino)
    for view_angle in range(np.shape(sino_filtered)[0]):
        sino_filtered[view_angle] = sgn.convolve(sino_filtered[view_angle], kernel, mode='same')
    else:
        sino_filtered = sino

    # Begin qGGMRF reconstruction

    print('Generating initial image with 3D qGGMRF reconstruction ...\n')

    init_image = mbircone.cone3D.recon(sino_filtered, angles, dist_source_detector, magnification,
                                          delta_pixel_image=delta_pixel*2,
                                          num_image_rows=num_image_rows//2, num_image_cols=num_image_cols//2,
                                          num_image_slices=num_image_slices//2,
                                          sharpness=sharpness, T=T, max_resolutions=0)

    # double size of pixels to return to original array size
    init_image = np.repeat(init_image, 2, axis=0)
    init_image = np.repeat(init_image, 2, axis=1)
    init_image = np.repeat(init_image, 2, axis=2)
else:
    init_image=0.0

######################################################################################
# Perform final qGGMRF reconstruction
######################################################################################

print('Performing 3D qGGMRF reconstruction ...\n')

recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, init_image=init_image,
                              delta_pixel_image=delta_pixel,
                              num_image_rows=num_image_rows, num_image_cols=num_image_cols,
                              num_image_slices=num_image_slices,
                              sharpness=sharpness, T=T, max_resolutions=0)

##################################################################################################
# Display phantom, synthetic sinogram, filtered sinogram, initial image, and reconstruction images
##################################################################################################
# Set display indexes for phantom and recon images
display_slice_phantom = num_phantom_slices // 2
display_x_phantom = num_phantom_rows // 2
display_y_phantom = num_phantom_cols // 2
display_slice_recon = num_image_slices // 2
display_x_recon = num_image_rows // 2
display_y_recon = num_image_cols // 2


# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))

if use_initial_image and filter_sinogram:
    # filtered sinogram images
    for view_idx in [0, num_views//4, num_views//2]:
        view_angle = int(angles[view_idx]*180/np.pi)
        plot_image(sino_filtered[view_idx, :, :], title=f'filtered sinogram view angle {view_angle} ',
                 filename=os.path.join(save_path, f'sino-filtered-shepp-logan-3D-view_angle{view_angle}.png'))

# display phantom images
plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)

# display recon images
plot_image(recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}',
          filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}',
          filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,:,display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}',
          filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)
          
if isinstance(init_image, np.ndarray) and (init_image.ndim == 3):

    if filter_sinogram:
        # display cross section of filter through one axis
        kernel_cross_section = kernel[order_kernel]
        x_ticks = [x for x in range(-order_kernel, order_kernel+1)]
        plt.clf()
        plt.plot(x_ticks, kernel_cross_section)
        plt.title(f'Cross section of convolutional kernel, sampling_rate={sampling_rate_kernel}')
        plt.xlabel('Index')
        plt.ylabel('Value of kernel at index')
        plt.savefig(os.path.join(save_path, f'kernel_cross_section.png'), format='png')
        plt.show()
  
    display_slice_init_image = num_image_slices // 2
    display_x_init_image = num_image_rows // 2
    display_y_init_image = num_image_cols // 2

    # display initial images
    plot_image(init_image[display_slice_init_image], title=f'qGGMRF init_image, axial slice {display_slice_init_image}',
             filename=os.path.join(save_path, 'init_image_axial.png'), vmin=vmin, vmax=vmax)
    plot_image(init_image[:,display_x_init_image,:], title=f'qGGMRF init_image, coronal slice {display_x_init_image}',
             filename=os.path.join(save_path, 'init_image_coronal.png'), vmin=vmin, vmax=vmax)
    plot_image(init_image[:,:,display_y_init_image], title=f'qGGMRF init_image, sagittal slice {display_y_init_image}',
             filename=os.path.join(save_path, 'init_image_sagittal.png'), vmin=vmin, vmax=vmax)

print(f"Images saved to {save_path}.")
input("Press Enter")
