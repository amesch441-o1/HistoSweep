import os
import argparse
from HistoSweep_single import run_histosweep_single

# ==================  USER CONFIGURATION ==================

HE_prefix = 'HE/demo/' # Path prefix to your H&E image folder
output_dir = f"{HE_prefix}/HistoSweep_Output"  #Folder for HistoSweep results

# Processing flags
need_scaling_flag = True 
need_preprocessing_flag = True 

# Image and pipeline parameters
pixel_size_raw = 0.5      # Microns per pixel in original image
pixel_size = 0.5          # Target pixel size (for rescaling, if needed)
patch_size = 16           # Size of each patch used in analysis
density_thresh = 100      # Density threshold
clean_background_flag = True
min_size = 10             # Parameter used to remove isolated debris and small specs outside tissue



#####################################################################################
#####################################################################################
#####################################################################################


user_config = {
    'HE_prefix': f"{HE_prefix}",
    'output_dir':  f"{output_dir}",
    'pixel_size_raw': pixel_size_raw,
    'density_thresh': density_thresh,
    'clean_background_flag': clean_background_flag,
    'min_size': min_size,
    'patch_size': patch_size,
    'pixel_size': pixel_size,
    'need_scaling_flag': need_scaling_flag,
    'need_preprocessing_flag': need_preprocessing_flag,
}



def str2bool(v):
    return str(v).lower() in ('true', '1', 'yes', 'y', 't')


def parse_args():
    parser = argparse.ArgumentParser(description="Run HistoSweep single image pipeline")

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--pixel_size_raw', type=float)
    parser.add_argument('--density_thresh', type=float)
    parser.add_argument('--clean_background_flag', type=str2bool)
    parser.add_argument('--min_size', type=int)
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--pixel_size', type=float)
    parser.add_argument('--need_scaling_flag', type=str2bool)
    parser.add_argument('--need_preprocessing_flag', type=str2bool)
    parser.add_argument('--HE_prefix', type=str)

    return parser.parse_args()


def main():
    print("\n******* Starting HistoSweep single image pipeline... *******")

    args = parse_args()

    # Merge user CLI args with defaults
    config = {k: getattr(args, k) if getattr(args, k) is not None else v
              for k, v in user_config.items()}

    os.makedirs(config['output_dir'], exist_ok=True)

    run_histosweep_single(
        output_base_dir=config['output_dir'],
        pixel_size_raw=config['pixel_size_raw'],
        density_thresh=config['density_thresh'],
        clean_background_flag=config['clean_background_flag'],
        min_size=config['min_size'],
        patch_size=config['patch_size'],
        pixel_size=config['pixel_size'],
        need_scaling_flag=config['need_scaling_flag'],
        need_preprocessing_flag=config['need_preprocessing_flag'],
        HE_prefix=config['HE_prefix']
    )

    print("✅ HistoSweep processing complete.")


if __name__ == "__main__":
    main()
