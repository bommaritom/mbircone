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
the number of iterations in qGGMRF reconstruction of the Shepp
Logan Phantom by providing an initial image. Experiment A is a regular 
qGGMRF reconstruction. Experiment B is a qGGMRF reconstruction using 
a reconstruction at half-pixel pitch as the initial image. Experiment C
is a qGGMRF reconstruction using a reconstruction at half-pixel pitch
generated from a sinogram filtered using a convolutional filter. Multiple
filters can be tested for Experiment C by adding the parameters to 'f_array'
and 'p_array'. It is recommended to run experiments separately, so as 
not to generate too many plots at once.
"""
print('\t The goal of this script is to determine whether we can reduce\
\n\t the number of iterations in qGGMRF reconstruction of the Shepp\
\n\t Logan Phantom by providing an initial image. Experiment A is a regular \
\n\t qGGMRF reconstruction. Experiment B is a qGGMRF reconstruction using \
\n\t a reconstruction at half-pixel pitch as the initial image. Experiment C\
\n\t is a qGGMRF reconstruction using a reconstruction at half-pixel pitch\
\n\t generated from a sinogram filtered using a convolutional filter. Multiple\
\n\t filters can be tested for Experiment C by adding the parameters to \'f_array\'\
\n\t and \'p_array\'. It is recommended to run experiments separately, so as \
\n\t not to generate too many plots at once.\n')


# #################################################################################
# Experiment Parameters
# #################################################################################

experiment_A = False                         # Toggles code corresponding to Experiment A, or regular
                                             # qGGMRF cone beam reconstruction without an initial image.

experiment_B = False                         # Toggles code corresponding to Experiment B, or qGGMRF cone
                                             # beam reconstruction with an initial image generated at
                                             # half-pixel pitch.

experiment_C = True                          # Toggles code corresponding to Experiment C, or qGGMRF cone
                                             # beam reconstruction with an initial image generated at
                                             # half-pixel pitch from a sinogram filtered according to the
                                             # kernels given below.

f_array = [2.0, 4.0, 6.0]                    # Sampling rate of 2D sinc function with Hamming window.
p_array = [4, 8. 12]                         # Order of convolutional kernel. Support of kernel
                                             # is 2*(order_kernel)+1
                                             # For each pair (f,p) provided, one full
#                                            # reconstruction is run.

# #################################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and recon
# #################################################################################


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


######################################################################################
# Generate a 3D shepp logan phantom
######################################################################################

print('Genrating 3D Shepp Logan phantom ...\n')

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
# Generate initial images by running recon on lower pixel pitch
######################################################################################

if experiment_C:
    # Generate convolutional filter to be applied to the sinogram.

    # Build kernels

    kernel_dict = dict()

    assert np.size(f_array) == np.size(p_array), 'Size of f_array and p_array must match.'

    num_kernels = np.size(f_array)

    for index in range(num_kernels):
        kernel_dict[index] = get_kernel(f_array[index], p_array[index])

    # Build filtered sinograms, for Experiment C

    sino_C_dict = dict()

    for index in range(num_kernels):
        kernel = kernel_dict[index]
        sino_C = np.copy(sino)

        # filter sinogram by convolving with kernel
        for view_angle in range(np.shape(sino_C)[0]):
            sino_C[view_angle] = sgn.convolve(sino_C[view_angle], kernel, mode='same')
            sino_C_dict[index] = sino_C

# Begin qGGMRF reconstruction

if experiment_B:
    # Generate initial image from unfiltered sinogram, for Experiment B

    print('Generating init_image_B with 3D qGGMRF reconstruction ...\n')

    init_image_B = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                             delta_pixel_image=delta_pixel*2,
                                             num_image_rows=num_image_rows//2, num_image_cols=num_image_cols//2,
                                             num_image_slices=num_image_slices//2,
                                             sharpness=sharpness, T=T, max_resolutions=0)

    # double size of pixels to return to original array size
    init_image_B = np.repeat(init_image_B, 2, axis=0)
    init_image_B = np.repeat(init_image_B, 2, axis=1)
    init_image_B = np.repeat(init_image_B, 2, axis=2)

if experiment_C:
    # Generate initial images from filtered sinograms, for Experiment C

    init_image_C_dict = dict()

    for index in range(num_kernels):

        sino_C = sino_C_dict[index]

        print(f'Generating init_image_C_{index} with 3D qGGMRF reconstruction ...\n')

        init_image_C = mbircone.cone3D.recon(sino_C, angles, dist_source_detector, magnification,
                                              delta_pixel_image=delta_pixel*2,
                                              num_image_rows=num_image_rows//2, num_image_cols=num_image_cols//2,
                                              num_image_slices=num_image_slices//2,
                                              sharpness=sharpness, T=T, max_resolutions=0)

        # Double size of pixels to return to original array size
        init_image_C = np.repeat(init_image_C, 2, axis=0)
        init_image_C = np.repeat(init_image_C, 2, axis=1)
        init_image_C = np.repeat(init_image_C, 2, axis=2)

        # Add initial image to array of sinograms

        init_image_C_dict[index] = init_image_C



######################################################################################
# Perform final qGGMRF reconstruction
######################################################################################

if experiment_A:

    # qGGMRF reconstruction, no initial image

    print('Generating recon_A with 3D qGGMRF reconstruction ...\n')

    recon_A = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                          delta_pixel_image=delta_pixel,
                                          num_image_rows=num_image_rows, num_image_cols=num_image_cols,
                                          num_image_slices=num_image_slices,
                                          sharpness=sharpness, T=T, max_resolutions=0)

if experiment_B:

    # qGGMRF reconstruction with initial image, unfiltered sinogram

    print('Generating recon_B with 3D qGGMRF reconstruction ...\n')

    recon_B = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                                       init_image=init_image_B,
                                                       delta_pixel_image=delta_pixel,
                                                       num_image_rows=num_image_rows, num_image_cols=num_image_cols,
                                                       num_image_slices=num_image_slices,
                                                       sharpness=sharpness, T=T, max_resolutions=0)

if experiment_C:

    # qGGMRF reconstruction with initial image, filtered sinograms

    recon_C_dict = dict()

    for index in range(num_kernels):

        init_image_C = init_image_C_dict[index]

        print(f'Generating init_image_C_{index} with 3D qGGMRF reconstruction ...\n')

        recon_C = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, init_image=init_image_C,
                                      delta_pixel_image=delta_pixel,
                                      num_image_rows=num_image_rows, num_image_cols=num_image_cols,
                                      num_image_slices=num_image_slices,
                                      sharpness=sharpness, T=T, max_resolutions=0)

        recon_C_dict[index] = recon_C


##################################################################################################
# Display phantom, synthetic sinogram, filtered sinogram, initial image, and reconstruction images
##################################################################################################

# Set display indexes for phantom and recon images
display_slice_phantom = num_phantom_slices // 2
display_x_phantom = num_phantom_rows // 2
display_y_phantom = num_phantom_cols // 2

display_slice_init_image = num_image_slices // 2
display_x_init_image = num_image_rows // 2
display_y_init_image = num_image_cols // 2

display_slice_recon = num_image_slices // 2
display_x_recon = num_image_rows // 2
display_y_recon = num_image_cols // 2


# Display phantom
plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)

# Display sino
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino_view_angle_{view_angle}.png'))

if experiment_A:

    # Display recon_A
    plot_image(recon_A[display_slice_recon], title=f'qGGMRF recon A, axial slice {display_slice_recon}',
              filename=os.path.join(save_path, 'recon_A_axial.png'), vmin=vmin, vmax=vmax)
    plot_image(recon_A[:,display_x_recon,:], title=f'qGGMRF recon A, coronal slice {display_x_recon}',
              filename=os.path.join(save_path, 'recon_A_coronal.png'), vmin=vmin, vmax=vmax)
    plot_image(recon_A[:,:,display_y_recon], title=f'qGGMRF recon A, sagittal slice {display_y_recon}',
              filename=os.path.join(save_path, 'recon_A_sagittal.png'), vmin=vmin, vmax=vmax)

if experiment_B:

    # Display init_image_B
    plot_image(init_image_B[display_slice_init_image], title=f'init_image_B from qGGMRF, axial slice {display_slice_init_image}',
             filename=os.path.join(save_path, 'init_image_B_axial.png'), vmin=vmin, vmax=vmax)
    plot_image(init_image_B[:,display_x_init_image,:], title=f'init_image_B from qGGMRF, coronal slice {display_x_init_image}',
             filename=os.path.join(save_path, 'init_image_B_coronal.png'), vmin=vmin, vmax=vmax)
    plot_image(init_image_B[:,:,display_y_init_image], title=f'init_image_B from qGGMRF, sagittal slice {display_y_init_image}',
             filename=os.path.join(save_path, 'init_image_B_sagittal.png'), vmin=vmin, vmax=vmax)

    # Display recon_B
    plot_image(recon_B[display_slice_recon], title=f'qGGMRF recon B from initial image, axial slice {display_slice_recon}',
              filename=os.path.join(save_path, 'recon_B_axial.png'), vmin=vmin, vmax=vmax)
    plot_image(recon_B[:,display_x_recon,:], title=f'qGGMRF recon B from initial image, coronal slice {display_x_recon}',
              filename=os.path.join(save_path, 'recon_B_coronal.png'), vmin=vmin, vmax=vmax)
    plot_image(recon_B[:,:,display_y_recon], title=f'qGGMRF recon B from initial image, sagittal slice {display_y_recon}',
              filename=os.path.join(save_path, 'recon_B_sagittal.png'), vmin=vmin, vmax=vmax)

if experiment_C:

    # Display each sino_C
    for index in range(num_kernels):
        sino_C = sino_C_dict[index]
        for view_idx in [0, num_views//4, num_views//2]:
            view_angle = int(angles[view_idx]*180/np.pi)
            plot_image(sino_C[view_idx, :, :],
                       title=f'filtered sinogram {index}, f={f_array[index]}, p={p_array[index]}, view angle {view_angle} ',
                       filename=os.path.join(save_path, f'sino_filtered_kernel_{index}_view_angle_{view_angle}.png'))

    # Display each init_image_C
    for index in range(num_kernels):

        init_image_C = init_image_C_dict[index]

        plot_image(init_image_C[display_slice_init_image],
                   title=f'init_image_C_{index} from qGGMRF, axial slice {display_slice_init_image}',
                   filename=os.path.join(save_path, f'init_image_C_kernel_{index}_axial.png'), vmin=vmin, vmax=vmax)
        plot_image(init_image_C[:,display_x_init_image,:],
                   title=f'init_image_C_{index} from qGGMRF, coronal slice {display_x_init_image}',
                   filename=os.path.join(save_path, f'init_image_C_kernel_{index}_coronal.png'), vmin=vmin, vmax=vmax)
        plot_image(init_image_C[:,:,display_y_init_image],
                   title=f'init_image_C_{index} from qGGMRF, sagittal slice {display_y_init_image}',
                   filename=os.path.join(save_path, f'init_image_C_kernel_{index}_sagittal.png'), vmin=vmin, vmax=vmax)

    # Display recon_C
    for index in range(num_kernels):

        recon_C = recon_C_dict[index]

        plot_image(recon_C[display_slice_recon],
                   title=f'qGGMRF recon C, f={f_array[index]}, p={p_array[index]}, axial slice {display_slice_recon}',
                   filename=os.path.join(save_path, f'recon_C_kernel_{index}_axial.png'), vmin=vmin, vmax=vmax)
        plot_image(recon_C[:, display_x_recon, :],
                   title=f'qGGMRF recon C, f={f_array[index]}, p={p_array[index]}, coronal slice {display_x_recon}',
                   filename=os.path.join(save_path, f'recon_C_kernel_{index}_coronal.png'), vmin=vmin, vmax=vmax)
        plot_image(recon_C[:, :, display_y_recon],
                   title=f'qGGMRF recon C, f={f_array[index]}, p={p_array[index]}, sagittal slice {display_y_recon}',
                   filename=os.path.join(save_path, f'recon_C_kernel_{index}_sagittal.png'), vmin=vmin, vmax=vmax)

    # Display cross section of filter through one axis
    for index in range(num_kernels):
        kernel = kernel_dict[index]

        # Retrieve a cross section through the origin, located at kernel[p]
        kernel_cross_section = kernel[p_array[index]]

        # Plot kernel
        x_ticks = [x for x in range(-p_array[index], p_array[index]+1)]
        plt.clf()
        plt.plot(x_ticks, kernel_cross_section)
        plt.title(f'Cross section of convolutional kernel, sampling_rate={f_array[index]}')
        plt.xlabel('Index')
        plt.ylabel('Value of kernel at index')
        plt.savefig(os.path.join(save_path, f'kernel_{index}_cross_section.png'), format='png')
        plt.show()


print(f"Images saved to {save_path}.")
input("Press Enter")
