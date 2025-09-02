#### Load package ####
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import pickle
import os
import tifffile
import pyvips
import glob
import matplotlib as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = 10e100
import tracemalloc
from functools import wraps

def measure_peak_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        current_gb = current / (1024 ** 3)
        peak_gb = peak / (1024 ** 3)
        print(f"[{func.__name__}] Current memory: {current_gb:.4f} GB; Peak memory: {peak_gb:.4f} GB")
        tracemalloc.stop()
        return result
    return wrapper


#### Basic functions ####

def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

def find_he(base_dir, stem):
    exts = ['.tiff', '.svs', '.ome.tif', '.tif', '.jpg', '.png','.ndpi']
    for ext in exts:
        candidate = os.path.join(base_dir, f"{stem}{ext}")
        #print(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def get_image_filename(prefix, printName = True):
    print("\nLooking for files with prefix:", prefix)
    #print("Current working directory:", os.getcwd())
    print("All matching files:", glob.glob(prefix + ".*"))
    
    for suffix in ['.jpg', '.png', '.tiff', '.tif', '.svs', '.ome.tif', '.ndpi']:
        filename = prefix + suffix
        if printName:
            print(f"Checking: {filename}")
        if os.path.exists(filename):
            print(f"✅ Found: {filename}")
            return filename
    raise FileNotFoundError(f'Image not found with prefix: {prefix}')



def smart_save_image(img, prefix, base_name="base", size_threshold=60000):
    """
    Save image as JPG if both dimensions are under `size_threshold`, otherwise as TIFF.
    """
    h, w = img.shape[:2]
    print(f"Image size: {h}x{w}")

    if h < size_threshold and w < size_threshold:
        # Save as JPG
        path = f"{prefix}{base_name}.jpg"
        Image.fromarray(img.astype(np.uint8)).save(path, quality=90)
        print(f"✅ Saved as JPG: {path}")
    else:
        # Save as TIFF
        path = f"{prefix}{base_name}.tiff"
        tifffile.imwrite(path, img, bigtiff=True)
        print(f"✅ Saved as TIFF: {path}")


def to_vips(img):
    """Convert NumPy or PyVIPS image to PyVIPS.Image."""
    if isinstance(img, pyvips.Image):
        return img
    elif isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 2:
            return pyvips.Image.new_from_memory(img.tobytes(), w, h, 1, 'uchar')
        elif img.ndim == 3 and img.shape[2] == 3:
            return pyvips.Image.new_from_memory(img.tobytes(), w, h, 3, 'uchar')
        else:
            raise ValueError("Unsupported NumPy array shape")
    else:
        raise TypeError("Input must be a NumPy array or pyvips.Image")

def smart_save_image_vips(img, prefix, base_name="base", size_threshold=60000):
    """
    Save image as JPG if both dimensions are under `size_threshold`, otherwise as TIFF.
    Accepts NumPy array or pyvips.Image.
    """
    vips_img = to_vips(img)
    h, w = vips_img.height, vips_img.width
    print(f"Image size: {h}x{w}")

    if h < size_threshold and w < size_threshold:
        path = f"{prefix}{base_name}.jpg"
        vips_img.jpegsave(path, Q=90)
        print(f"✅ Saved as JPG: {path}")
    else:
        path = f"{prefix}{base_name}.tiff"
        vips_img.tiffsave(path, bigtiff=True, compression="lzw", tile=True, pyramid=True)
        print(f"✅ Saved as TIFF: {path}")




def read_string(filename):
    return read_lines(filename)[0]


# def write_string(string, filename):
#     return write_lines([string], filename)


# def load_image(filename, verbose=True):
#     img = Image.open(filename)
#     img = np.array(img)
#     if img.ndim == 3 and img.shape[-1] == 4:
#         img = img[..., :3]  # remove alpha channel
#     if verbose:
#         print(f'Image loaded from {filename}')
#     return img


# modified raw load_image
def load_image(filename, verbose=True):
    ext = os.path.splitext(filename)[-1].lower()
    print(f"\nfilename = {filename},ext={ext}")
    # use tifffile to open .tif / .tiff
    if ext in [".svs"]:
        print("this is a svs file......")
        import pyvips
        # Load the SVS file
        slide = pyvips.Image.new_from_file(filename, access='random')
        # Get image properties
        print(f"Width: {slide.width}, Height: {slide.height}, Bands: {slide.bands}")
        # Convert to NumPy array
        img = np.ndarray(
            buffer=slide.write_to_memory(),
            dtype=np.uint8,  # assuming 8-bit image
            shape=(slide.height, slide.width, slide.bands)
        )
        print(img.shape)
    if ext in ['.tif', '.tiff']:
        print("this is tif|tiff file......")
        img = tifffile.imread(filename)
        img = np.array(img)
    if ext in ['.jpg', '.jpeg', '.png']:
        print("this is jpg|jpeg|png file......")
        img = Image.open(filename)
        img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img

def load_image_vips(filename):
    print("read image with pyvips.....")
    image = pyvips.Image.new_from_file(filename, access='random')
    print(f"Image Width: {image.width}, Image Height: {image.height}, Image Bands: {image.bands}")
    return image


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df


def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)
    print(filename)

        
def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)


def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask

    
def save_colormapped_map(map_norm, title, filename, output_dir):
    colormap = plt.get_cmap("jet")
    colored = colormap(map_norm)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    Image.fromarray(colored).save(os.path.join(output_dir, filename))
    print(f"\n {title} map saved as '{filename}'")
