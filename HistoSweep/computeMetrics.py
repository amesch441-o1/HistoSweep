#!/usr/bin/env python3
import numpy as np
import gc
import time
from functools import wraps
from numba import jit, prange
from utils import measure_peak_memory

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} costs: {end - start:.4f}s")
        return result
    return wrapper

def get_memory_usage():
    """get current memory usage in GB"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024**3)  # GB
    except ImportError:
        return None

# Normalize an array to the range [1.0, 2.0]
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def normalize_array(arr):
    n = len(arr)
    result = np.empty(n, dtype=arr.dtype)
    min_val = arr[0]
    max_val = arr[0]
    for i in range(1, n):
        val = arr[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    range_val = max_val - min_val
    if range_val > 0:
        inv_range = 1.0 / range_val
        for i in prange(n):
            result[i] = (arr[i] - min_val) * inv_range + 1.0
    else:
        result[:] = 1.0
    return result

# patchify with reshape(np.uint8)
def patchify_reshape_lazy(x, patch_size):
    h, w, c = x.shape
    tiles_shape = (h // patch_size, w // patch_size)
    patches = x.reshape(
        tiles_shape[0], patch_size,
        tiles_shape[1], patch_size, c
    ).transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, c)
    
    shapes = dict(tiles=np.array(tiles_shape))
    return patches, shapes

# calculate std and mean in chunks
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_std_and_mean_chunk_lazy(patches_uint8):
    n_patches, h, w, c = patches_uint8.shape
    stds = np.empty(n_patches, dtype=np.float32)
    means = np.empty((n_patches, c), dtype=np.float32)
    
    hwc = h * w * c
    hw = h * w
    
    for i in prange(n_patches):
        patch_uint8 = patches_uint8[i]
        total = 0.0
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    total += float(patch_uint8[y, x, ch])  
        
        mean = total / hwc  
        # calculate variance across each chnnel
        var = 0.0
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    val = float(patch_uint8[y, x, ch])  # 临时转换
                    diff = val - mean
                    var += diff * diff
        
        stds[i] = np.sqrt(var / hwc)
        
        # calculate mean for each channel
        for ch in range(c):
            ch_sum = 0.0
            for y in range(h):
                for x in range(w):
                    ch_sum += float(patch_uint8[y, x, ch])  # temporary conversion
            means[i, ch] = ch_sum / hw
    
    return stds, means

# calculte vairance weighted mean (z_v)
@jit(nopython=True, fastmath=True, cache=True)
def compute_zv(means):
    n, c = means.shape
    ch_means = np.zeros(c, dtype=means.dtype)
    for ch in range(c):
        s = 0.0
        for i in range(n):
            s += means[i, ch]
        ch_means[ch] = s / n
    ch_vars = np.zeros(c, dtype=means.dtype)
    for ch in range(c):
        s = 0.0
        for i in range(n):
            diff = means[i, ch] - ch_means[ch]
            s += diff * diff
        ch_vars[ch] = s / n
    denom = ch_vars.sum()
    zv = np.zeros(n, dtype=means.dtype)
    for i in range(n):
        num = 0.0
        for ch in range(c):
            num += means[i, ch] * ch_vars[ch]
        zv[i] = num / denom
    return zv

@timer
@measure_peak_memory
def compute_metrics_fast(he, patch_size=16, chunk_size=200000):
    """
    """
    he_tiles, shapes = patchify_reshape_lazy(he, patch_size)    
    n_total = he_tiles.shape[0]
    c = he_tiles.shape[-1]
    
    stds_all = np.empty(n_total, dtype=np.float32)
    means_all = np.empty((n_total, c), dtype=np.float32)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    
    for chunk_idx, i in enumerate(range(0, n_total, chunk_size)):
        j = min(i + chunk_size, n_total)
        current_chunk_size = j - i
        
        if n_chunks > 1:
            print(f"\n  Chunk {chunk_idx+1}/{n_chunks}: {current_chunk_size:,} patches")
        chunk_uint8 = he_tiles[i:j]
        stds_chunk, means_chunk = compute_std_and_mean_chunk_lazy(chunk_uint8)
        
        # store result
        stds_all[i:j] = stds_chunk
        means_all[i:j] = means_chunk
        
        # clean memory
        del chunk_uint8, stds_chunk, means_chunk
        gc.collect()
    
    # delete he_tiles to free memory
    del he_tiles
    gc.collect()
    
    # reshape stds_all to image shape
    he_std_image = stds_all.reshape(shapes['tiles'])
    
    # calculate z_v
    z_v = compute_zv(means_all)
    del means_all
    gc.collect()
    z_v_image = z_v.reshape(shapes['tiles'])
    # Normalize stds_all and z_v
    std_norm = normalize_array(stds_all)
    z_v_norm = normalize_array(z_v)
    del stds_all, z_v
    gc.collect()
    # compute ratio
    ratio = std_norm / z_v_norm
    ratio_norm = normalize_array(ratio) - 1.0
    ratio_norm_image = ratio_norm.reshape(shapes['tiles'])
        
    return (
        std_norm.reshape(shapes['tiles']),
        he_std_image,
        z_v_norm.reshape(shapes['tiles']),
        z_v_image,
        ratio_norm,
        ratio_norm_image
    )


