#### Load package ####
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from PIL import Image
from utils import save_pickle
from utils import measure_peak_memory

@measure_peak_memory
def generate_final_mask(prefix, he, mask1_updated, mask2, output_dir, clean_background=True, super_pixel_size=16, minSize=10):

    masked = (mask1_updated.flatten() | mask2.flatten())
    image_height, image_width = he.shape[:2]

    num_super_pixels_y = image_height // super_pixel_size
    num_super_pixels_x = image_width // super_pixel_size
    mask = masked.reshape((num_super_pixels_y, num_super_pixels_x))
    cleaned = 1 - mask
    cleaned = (cleaned * 255).astype(np.uint8)

    if clean_background:
        binary = 1 - mask
        cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=minSize, connectivity=2)
        cleaned = (cleaned * 255).astype(np.uint8)

    os.makedirs(output_dir, exist_ok=True)

    # Save cleaned mask (super-pixel level)
    Image.fromarray(cleaned).save(os.path.join(output_dir, 'mask-small.png'))

    # Upscale to full resolution
    super_pixel_values = cleaned == 0
    mask_final = np.kron(~super_pixel_values, np.ones((super_pixel_size, super_pixel_size), dtype=np.uint8)) * 255

    Image.fromarray(mask_final).save(os.path.join(output_dir, 'mask.png'))

    print("\n np.sum(mask_final):", np.sum(mask_final))
    print("np.sum(cleaned):", np.sum(cleaned))
    print("✅ Final masks saved in:", output_dir)
