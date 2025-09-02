# ===== USER-DEFINED INPUT PARAMETERS =====

# Path prefix to your H&E image folder
HE_prefix = 'HE/H1_high/'
# Directory for output 
output_directory = "HistoSweep_Output" #Folder for HistoSweep output/results
# Flag for whether to rescale the image 
need_scaling_flag = False  # True if image resolution ≠ 0.5µm (or desired size) per pixel
# Flag for whether to preprocess the image 
need_preprocessing_flag = False  # True if image dimensions are not divisible by patch_size
# The pixel size (in microns) of the raw H&E image 
pixel_size_raw = 0.5  # Typically provided by the scanner/metadata (e.g., 0.25 µm/pixel for 40x)
# Parameter used determine amount of density filtering (e.g artifacts) (consider lowering for VERY large images)
density_thresh = 100 # Typically 100 works well, but may need to increase if artifacts are not being effectively removed (e.g. fiducial marker)
# Flag for whether to clean background (i.e. remove isolated debris and small specs outside tissue)
clean_background_flag = True # Set to False if you want to preserve fibrous regions that are otherwise being incorrectly filtered out
# Parameter used to remove isolated debris and small specs outside tissue
min_size = 10 # Decrease if there are many fibrous areas (e.g. adipose) in the tissue that you wish to retain (e.g. 5), increase if lots of/larger debris you wish to remove (e.g.50)
# ===== Additional PARAMETERS (typically do not need to change) =====
# Size of one square patch (superpixel) used throughout processing
patch_size = 16  # 16x16 pixels → typically 8µm if pixel_size = 0.5
# Target pixel size (in microns)
pixel_size = 0.5  # Final desired resolution; keep as 0.5 µm for standardization

import os
from utils import load_image, get_image_filename
from saveParameters import saveParams
#from computeMetrics import compute_metrics
from computeMetrics import compute_metrics_fast
from densityFiltering import compute_low_density_mask
#from textureAnalysis import run_texture_analysis
from textureAnalysis import run_texture_analysis_optimized
from ratioFiltering import run_ratio_filtering
from generateMask import generate_final_mask
from additionalPlots import generate_additionalPlots
import numpy as np 
image = load_image(get_image_filename(HE_prefix+'he'))
print(image.shape)

if not os.path.exists(f"{HE_prefix}{output_directory}"):
    os.makedirs(f"{HE_prefix}{output_directory}")

saveParams(HE_prefix, output_directory, need_scaling_flag, need_preprocessing_flag, pixel_size_raw,density_thresh,clean_background_flag,min_size,patch_size,pixel_size)

he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics_fast(image, patch_size=patch_size, chunk_size=50000) 

# identify low density superpixels
mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
#np.save(f"{HE_prefix}{output_directory}/mask1_lowdensity.npy", mask1_lowdensity)
print('Total selected for density filtering: ', mask1_lowdensity.sum()) #Total selected for density filtering:  8079

# perform texture analysis 
mask1_lowdensity_update = run_texture_analysis_optimized(prefix=HE_prefix, image=image, tissue_mask=mask1_lowdensity, output_dir=output_directory, patch_size=patch_size, glcm_levels=64)

# identify low ratio superpixels
mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)

print(mask2_lowratio.shape)
print("otsu threshold: ", otsu_thresh)
print("mask2_lowratio sum: ", mask2_lowratio.sum())
generate_final_mask(prefix=HE_prefix, he=image, output_dir = f"{HE_prefix}{output_directory}",
                    mask1_updated = mask1_lowdensity_update, mask2 = mask2_lowratio,
                    clean_background = clean_background_flag, 
                    super_pixel_size=patch_size, minSize = min_size)