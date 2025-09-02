def run_histosweep_for_image(image_path, output_base_dir, pixel_size_raw=0.5, density_thresh=100,
                             clean_background_flag=True, min_size=10, patch_size=16, pixel_size=0.5,
                             need_scaling_flag=True, need_preprocessing_flag=True):

    import os
    import os.path as osp
    import subprocess
    from utils import load_image
    from saveParameters import saveParams
    from computeMetrics import compute_metrics_fast
    from densityFiltering import compute_low_density_mask
    from textureAnalysis import run_texture_analysis_optimized
    from ratioFiltering import run_ratio_filtering
    from generateMask import generate_final_mask
    from additionalPlots import generate_additionalPlots
    import glob

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_directory = osp.join(output_base_dir, f"HistoSweep_Output_{image_name}")


    os.makedirs(output_directory, exist_ok=True)

    # *** Here's the fix: get prefix = full path without extension ***
    image_prefix = os.path.splitext(image_path)[0]

    print(f"\nProcessing image: {image_name}")
    print(f"📁 Output dir: {output_directory}")

    # ----- Optional Preprocessing -----
    if need_scaling_flag:
        #print(f"\n ***Rescaling {image_name} ...***")
        subprocess.run([
            "python", "rescale.py", "--image",
            "--prefix", image_prefix,
            "--pixelSizeRaw", str(pixel_size_raw),
            "--pixelSize", str(pixel_size),
            "--filename", os.path.basename(image_path),
            "--outputDir", output_directory
        ], check=True)

    if need_preprocessing_flag:
       # print(f"\n *** Preprocessing {image_name} ...***")
        subprocess.run([
            "python", "preprocess.py", "--image",
            "--prefix", image_prefix,
            "--pixelSizeRaw", str(pixel_size_raw),
            "--pixelSize", str(pixel_size),
            "--filename", os.path.basename(image_path),
            "--outputDir", output_directory
        ], check=True)

    # ----- Load Image -----
    #image = load_image(image_path)
    #print(f"\nLoaded image shape: {image.shape}")

    # ----- Load Image -----
    # Priority: he → he-scaled → original
    he_img = None
    exts = ['.tiff', '.svs', '.ome.tif', '.jpg', '.png', '.ndpi']  # Add any others you expect

    for ext in exts:
        candidate = os.path.join(output_directory, f"he{ext}")
        if os.path.exists(candidate):
            he_img = candidate
            print(f" Using image: {he_img}")
            break

    if he_img is None:
        for ext in exts:
            candidate = os.path.join(output_directory, f"he-scaled{ext}")
            if os.path.exists(candidate):
                he_img = candidate
                print(f" Using image: {he_img}")
                break

    if he_img is None:
        he_img = image_path
        print(f" Using image: {he_img}")

    image = load_image(he_img)
    print(f"\nLoaded image shape: {image.shape}")


    # ----- Save Parameters -----
    saveParams(image_prefix, output_directory, need_scaling_flag, need_preprocessing_flag,
               pixel_size_raw, density_thresh, clean_background_flag, min_size, patch_size, pixel_size)

    # ----- HistoSweep Pipeline -----
    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics_fast(
        image, patch_size=patch_size)

    mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)

    mask1_lowdensity_update = run_texture_analysis_optimized(
        prefix=image_path,
        image=image,
        tissue_mask=mask1_lowdensity,
        output_dir=output_directory,
        patch_size=patch_size,
        glcm_levels=64
    )

    mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)

    generate_final_mask(
        prefix=image_prefix,
        he=image,
        mask1_updated=mask1_lowdensity_update,
        mask2=mask2_lowratio,
        output_dir=output_directory,
        clean_background=clean_background_flag,
        super_pixel_size=patch_size,
        minSize=min_size
    )

    #generate_additionalPlots(
    #    prefix=image_prefix,
    #    he=image,
    #    he_std_image=he_std_image_,
    #    he_std_norm_image=he_std_norm_image_,
    #    z_v_image=z_v_image_,
    #    z_v_norm_image=z_v_norm_image_,
    #    ratio_norm=ratio_norm_,
    #    ratio_norm_image=ratio_norm_image_,
    #    mask1=mask1_lowdensity,
    #    mask1_updated=mask1_lowdensity_update,
    #    mask2=mask2_lowratio,
    #    super_pixel_size=patch_size,
    #    output_dir=output_directory,
    #    generate_masked_plots=False
    #)

    print(f"\n Finished Image: {image_name}")
