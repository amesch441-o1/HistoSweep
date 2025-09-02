def run_histosweep_single(image_path, output_base_dir, pixel_size_raw=0.5, density_thresh=100,
                          clean_background_flag=True, min_size=10, patch_size=16, pixel_size=0.5,
                          need_scaling_flag=True, need_preprocessing_flag=True, HE_prefix = None):

    import os
    import os.path as osp
    import subprocess
    from utils import load_image, find_he
    from saveParameters import saveParams
    from computeMetrics import compute_metrics_fast
    from densityFiltering import compute_low_density_mask
    from textureAnalysis import run_texture_analysis_optimized
    from ratioFiltering import run_ratio_filtering
    from generateMask import generate_final_mask
    import glob

    #image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(image_path)
    output_dir = output_base_dir  # No image subdir in single-image mode.

    os.makedirs(output_dir, exist_ok=True)
    image_prefix = os.path.splitext(image_path)[0]

    print(f"\n📁 Processing image: ")
    print(f" Output dir: {output_dir}")

    # rescale and preprocess image 
    if need_scaling_flag:
        subprocess.run([
            "python", "rescale.py", "--image",
            "--prefix", HE_prefix,
            "--pixelSizeRaw", str(pixel_size_raw),
            "--pixelSize", str(pixel_size),
            "--outputDir", output_dir,
        ], check=True)
    else:
        print(" *Skipping rescale (need_scaling_flag=False).")

    if need_preprocessing_flag:
        subprocess.run([
            "python", "preprocess.py", "--image",
            "--prefix", HE_prefix,
            "--patchSize", str(patch_size),
            "--outputDir", output_dir,
            "--pixelSizeRaw", str(pixel_size_raw),
            "--pixelSize", str(pixel_size),
        ], check=True)
    else:
        print(" *Skipping preprocess (need_preprocessing_flag=False).")

    # ----- Load Final Preprocessed Image  -----

    he_img = (find_he(output_dir, 'he') or
            find_he(HE_prefix, 'he'))

    if he_img is None:
        raise FileNotFoundError(
            f"No usable H&E image found. Checked for 'he.*' in:\n"
            f"  {output_dir}\n  {HE_prefix}"
        )

    print(f"✅ Using image: {he_img}")
    image = load_image(he_img)
    print(f" Loaded image shape: {image.shape}")

    # ----- Save Parameters -----
    saveParams(image_prefix, output_dir, need_scaling_flag, need_preprocessing_flag,
               pixel_size_raw, density_thresh, clean_background_flag, min_size, patch_size, pixel_size)

    # ----- Run HistoSweep Pipeline -----
    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics_fast(
        image, patch_size=patch_size)

    mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)

    mask1_lowdensity_update = run_texture_analysis_optimized(
        prefix=image_path,
        image=image,
        tissue_mask=mask1_lowdensity,
        output_dir=output_dir,
        patch_size=patch_size,
        glcm_levels=64
    )

    mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)

    generate_final_mask(
        prefix=image_prefix,
        he=image,
        mask1_updated=mask1_lowdensity_update,
        mask2=mask2_lowratio,
        output_dir=output_dir,
        clean_background=clean_background_flag,
        super_pixel_size=patch_size,
        minSize=min_size
    )

    print(f"✅ Finished HistoSweep")
