import os
import argparse
from multiprocessing import Pool

# main per-image function
from HistoSweep_multi import run_histosweep_for_image

# ==================  USER CONFIGURATION (defaults) ==================

HE_prefix = 'HE/demo_multi/'          # Folder containing images
output_dir = f"{HE_prefix}/HistoSweep_Output"  # Output root

# Multi mode: parallel workers
num_workers = 4

# Required-for-multi processing flags
need_scaling_flag = True
need_preprocessing_flag = True

# Image & pipeline parameters
pixel_size_raw = 0.5
pixel_size = 0.5
patch_size = 16
density_thresh = 100
clean_background_flag = True
min_size = 10



#####################################################################################
#####################################################################################
#####################################################################################


# Accepted image formats
file_extensions = [".png", ".jpg", ".tif", ".svs", ".ome.tif", ".tiff", '.ndpi']

# Bundle defaults (so CLI can override any of them)
user_config = {
    'HE_prefix': HE_prefix,
    'output_dir': output_dir,
    'num_workers': num_workers,
    'pixel_size_raw': pixel_size_raw,
    'density_thresh': density_thresh,
    'clean_background_flag': clean_background_flag,
    'min_size': min_size,
    'patch_size': patch_size,
    'pixel_size': pixel_size,
    'need_scaling_flag': need_scaling_flag,
    'need_preprocessing_flag': need_preprocessing_flag,
}

# ================== CLI handling (optional overrides) ==================

def str2bool(v):
    return str(v).lower() in ('true', '1', 'yes', 'y', 't')

def parse_args():
    p = argparse.ArgumentParser(description="Run HistoSweep multi-image pipeline")
    p.add_argument('--HE_prefix', type=str, help="Input directory with images")
    p.add_argument('--output_dir', type=str, help="Output directory")
    p.add_argument('--num_workers', type=int, help="Parallel workers")
    p.add_argument('--pixel_size_raw', type=float)
    p.add_argument('--density_thresh', type=float)
    p.add_argument('--clean_background_flag', type=str2bool)
    p.add_argument('--min_size', type=int)
    p.add_argument('--patch_size', type=int)
    p.add_argument('--pixel_size', type=float)
    p.add_argument('--need_scaling_flag', type=str2bool)
    p.add_argument('--need_preprocessing_flag', type=str2bool)
    return p.parse_args()

# ================== Helpers ==================

def is_image_file(name: str) -> bool:
    return (not name.startswith("._")) and any(name.lower().endswith(ext) for ext in file_extensions)

# ================== Main ==================

def main():
    print("\n******* Starting HistoSweep MULTI image pipeline... *******")

    args = parse_args()

    # Merge CLI -> defaults
    config = {k: (getattr(args, k) if getattr(args, k) is not None else v)
              for k, v in user_config.items()}

    in_dir = config['HE_prefix']
    out_dir = config['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    # Strongly recommend these remain True for multi
    if not config['need_scaling_flag'] or not config['need_preprocessing_flag']:
        print("⚠️ For multi mode, need_scaling_flag and need_preprocessing_flag should be True. Running on raw image.")

    # Gather images
    all_images = [
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if is_image_file(f)
    ]
    print(f" Input dir: {in_dir}")
    print(f" Found {len(all_images)} compatible image(s).")
    print(f" Output dir: {out_dir}")
    print(f"  Workers: {config['num_workers']}")

    if not all_images:
        print("❌ No images found. Exiting.")
        return

    # Run in parallel
    with Pool(processes=config['num_workers']) as pool:
        pool.starmap(
            run_histosweep_for_image,
            [
                (
                    img_path,
                    out_dir,
                    config['pixel_size_raw'],
                    config['density_thresh'],
                    config['clean_background_flag'],
                    config['min_size'],
                    config['patch_size'],
                    config['pixel_size'],
                    config['need_scaling_flag'],
                    config['need_preprocessing_flag'],
                )
                for img_path in all_images
            ]
        )

    print("✅ HistoSweep MULTI processing complete.")

if __name__ == "__main__":
    main()
