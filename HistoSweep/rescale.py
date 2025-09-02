import argparse
import os
from time import time
from PIL import Image
import PIL
from utils import smart_save_image_vips, load_image_vips, get_image_filename

Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = 10e100

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--pixelSizeRaw', type=float, default=None)
    parser.add_argument('--pixelSize', type=float, default=0.5)
    parser.add_argument('--filename', type=str, default=None,
                        help='If provided, use this exact filename instead of the default he-raw.*')
    parser.add_argument('--outputDir', type=str, required=True,
                        help='Full path to the output folder for this image')
    return parser.parse_args()

def rescale_image_vips(img, scale):
    if scale == 1.0:
        return img
    img_rescaled = img.resize(scale)
    print(f"After rescaling, image Width: {img_rescaled.width}, image Height: {img_rescaled.height}")
    return img_rescaled 

def main():
    args = get_args()

    if args.pixelSizeRaw is None:
        raise ValueError("pixelSizeRaw must be provided")

    pixel_size_raw = args.pixelSizeRaw
    pixel_size = args.pixelSize
    scale = pixel_size_raw / pixel_size

    if args.image:
        if args.filename:
            # Multi-image mode
            img_dir = os.path.dirname(args.prefix)
            img_path = os.path.join(img_dir, args.filename)
        else:
            # Single-image mode
            img_path = get_image_filename(args.prefix + 'he-raw')

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")

        print(f"\nLoading image from: {img_path}")
        img = load_image_vips(img_path)

        print(f"\n ***Rescaling image (scale: {scale:.3f})...***")
        t0 = time()
        img = rescale_image_vips(img, scale)
        print(f"Rescaling took {int(time() - t0)} sec")
        print(f"Image size: {img.width} x {img.height}")

        # Save to the specified output directory
        output_dir = args.outputDir
        os.makedirs(output_dir, exist_ok=True)

        smart_save_image_vips(img, os.path.join(output_dir, ""), base_name="he-scaled", size_threshold=10000)

if __name__ == '__main__':
    main()
