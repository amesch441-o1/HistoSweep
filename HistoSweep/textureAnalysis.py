#!/usr/bin/env python3
import numpy as np
from numba import jit, prange, njit
import time
import os
import pandas as pd
from sklearn.mixture import GaussianMixture
from utils import measure_peak_memory, save_colormapped_map



@measure_peak_memory
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_all_texture_features_perfect(gray_image, positions, patch_size=16, levels=64):
    n_positions = positions.shape[0]
    features = np.zeros((n_positions, 4), dtype=np.float64)
    for idx in prange(n_positions):
        i, j = positions[idx, 0], positions[idx, 1]
        start_i = i * patch_size
        end_i = start_i + patch_size
        start_j = j * patch_size
        end_j = start_j + patch_size
        
        # calculate GLCM
        glcm = np.zeros((levels, levels), dtype=np.float64)
        
        # construct GLCM
        for pi in range(start_i, end_i):
            for pj in range(start_j, end_j - 1):
                pixel1 = min(max(gray_image[pi, pj], 0), levels - 1)
                pixel2 = min(max(gray_image[pi, pj + 1], 0), levels - 1)
                glcm[pixel1, pixel2] += 1
        
        # symmetrize GLCM 
        glcm = glcm + glcm.T
        total = np.sum(glcm)
        if total > 0:
            glcm = glcm / total
        
        # === the computing of GLCM features===
        energy_sum = 0.0
        homogeneity = 0.0
        entropy = 0.0
        for gi in range(levels):
            for gj in range(levels):
                p = glcm[gi, gj]
                energy_sum += p * p
                
                if p > 0:
                    # Homogeneity
                    homogeneity += p / (1.0 + (gi - gj) * (gi - gj))
                    
                    # Entropy
                    entropy -= p * np.log2(p)
        energy = np.sqrt(energy_sum)
                
        # save result
        features[idx, 0] = energy
        features[idx, 1] = homogeneity
        features[idx, 2] = entropy
    
    return features

# rgb2gray_jit
@njit(parallel=True)
def rgb2gray_jit(image):
    """convert RGB image to grayscale using JIT compilation"""
    h, w = image.shape[:2]
    gray = np.empty((h, w), dtype=np.uint8)
    
    for i in prange(h):
        for j in range(w):
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            gray_val = r * 0.2125 + g * 0.7154 + b * 0.0721
            gray[i, j] = np.uint8(gray_val)
    
    return gray

@jit(nopython=True, parallel=True)
def compute_is_filtered(image, positions, patch_size):
    n_positions = len(positions)
    is_filtered = np.zeros(n_positions, dtype=np.bool_)
    
    for k in prange(n_positions):
        i, j = positions[k, 0], positions[k, 1]
        start_i = i * patch_size
        end_i = start_i + patch_size
        start_j = j * patch_size
        end_j = start_j + patch_size
        sum_r = 0.0
        sum_g = 0.0
        sum_b = 0.0
        pixel_count = patch_size * patch_size
        for pi in range(start_i, end_i):
            for pj in range(start_j, end_j):
                sum_r += image[pi, pj, 0]
                sum_g += image[pi, pj, 1]
                sum_b += image[pi, pj, 2]
        
        mean_r = sum_r / pixel_count
        mean_g = sum_g / pixel_count
        mean_b = sum_b / pixel_count
        is_green = (mean_g > mean_r + 20) and (mean_g > mean_b + 20)
        rgb_values = np.array([mean_r, mean_g, mean_b])
        rgb_mean = (mean_r + mean_g + mean_b) / 3.0
        variance = ((mean_r - rgb_mean)**2 + (mean_g - rgb_mean)**2 + (mean_b - rgb_mean)**2) / 3.0
        rgb_std = np.sqrt(variance)
        is_gray = rgb_std < 10
        is_too_bright = (mean_r > 230) and (mean_g > 230) and (mean_b > 230)
        if is_green or is_gray or is_too_bright:
            is_filtered[k] = True
        else:
            is_filtered[k] = False
    
    return is_filtered

def measure_peak_memory(func):
    return func

@measure_peak_memory
def run_texture_analysis_optimized(prefix, image, tissue_mask, output_dir, patch_size=16, glcm_levels=64):
    
    output = os.path.join(output_dir, "AdditionalPlots", "textureAnalysis_plots")
    os.makedirs(output, exist_ok=True)
    t0 = time.time()
    print("Converting image to grayscale...")
    gray_image = rgb2gray_jit(image)
    t1 = time.time()
    print("*"*50)
    print(f"RGB-gray converts timt: {t1 - t0:.2f} seconds")
    print("*"*50)
    mask = tissue_mask.astype(bool)
    h, w = gray_image.shape
    h_mask, w_mask = mask.shape
    assert h // patch_size == h_mask and w // patch_size == w_mask, "Mask dimensions do not match superpixel grid"

    gray_image = (gray_image / 255 * (glcm_levels - 1)).astype(np.uint8)

    if mask.sum() == 0:
        print("✅ Skipping texture analysis — no low density superpixels.")
        return mask.copy()
    energy_map = np.full(mask.shape, np.nan)
    homogeneity_map = np.full(mask.shape, np.nan)
    entropy_map = np.full(mask.shape, np.nan)
    sharpness_map = np.full(mask.shape, np.nan)
    color_filter_mask = np.zeros(mask.shape, dtype=bool)
    t00 = time.time()
    print("Running ultra-fast texture analysis...")

    positions = np.argwhere(mask)  # shape = (N, 2)
    N = len(positions)

    is_filtered = compute_is_filtered(image, positions.astype(np.int32), patch_size)
 

    tracker = 0
    for k, (i, j) in enumerate(positions):
        if is_filtered[k]:
            color_filter_mask[i, j] = True
            tracker += 1

    positions_valid = positions[~is_filtered]

    if len(positions_valid) > 0:
        # warm up
        if len(positions_valid) > 5:
            _ = compute_all_texture_features_perfect(
                gray_image, positions_valid[:1].astype(np.int32), patch_size, glcm_levels
            )
        
    
        all_features = compute_all_texture_features_perfect(
            gray_image, positions_valid.astype(np.int32), patch_size, glcm_levels
        )
        
        energy_vals = all_features[:, 0]
        homogeneity_vals = all_features[:, 1]
        entropy_vals = all_features[:, 2]
        for idx, (i, j) in enumerate(positions_valid):
            energy_map[i, j] = energy_vals[idx]
            homogeneity_map[i, j] = homogeneity_vals[idx]
            entropy_map[i, j] = entropy_vals[idx]

    if len(positions_valid) < 2:
        print("✅ Skipping GMM clustering — insufficient valid data.")
        return mask.copy()

    energy_map_norm = (energy_map - np.nanmin(energy_map)) / (np.nanmax(energy_map) - np.nanmin(energy_map))
    homogeneity_map_norm = (homogeneity_map - np.nanmin(homogeneity_map)) / (np.nanmax(homogeneity_map) - np.nanmin(homogeneity_map))
    entropy_map_norm = (entropy_map - np.nanmin(entropy_map)) / (np.nanmax(entropy_map) - np.nanmin(entropy_map))

    features_df = pd.DataFrame({
        'homogeneity': homogeneity_map_norm.flatten(),
        'energy': energy_map_norm.flatten(),
        'entropy': entropy_map_norm.flatten(),
    })
    
    save_colormapped_map(entropy_map_norm, "Entropy", "glcm_entropy_map_colored.png", output)
    save_colormapped_map(energy_map_norm, "Energy", "glcm_energy_map_colored.png",output)
    save_colormapped_map(homogeneity_map_norm, "Homogeneity", "glcm_homogeneity_map_colored.png",output)

    print("Running GMM clustering...")
    valid_features = features_df.dropna().reset_index(drop=True)
    gmm = GaussianMixture(n_components=4, random_state=45)
    labels = gmm.fit_predict(valid_features)

    
    cluster_map = np.full(mask.shape, np.nan)
    valid_mask = ~np.isnan(homogeneity_map_norm)
    cluster_map[valid_mask] = labels

    means = valid_features.groupby(labels).mean()
    print("\n=== GLCM Metric Means ===")
    print(means)

    # keep clusters with lowest energy and homogeneity, highest entropy
    scores = means['energy'] + means['homogeneity'] - means['entropy']
    keep_labels = scores.nsmallest(2).index.tolist()
    keep_coords = np.where(np.isin(cluster_map, keep_labels))

    updated_mask = mask.copy()
    updated_mask[keep_coords] = False

    return updated_mask