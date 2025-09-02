<p align="left">
  <img src="assets/HistoSweepLogo.png" width="400"/>
</p>

# HistoSweep: Fast, Flexible H&E Image Quality Filtering at Scale

**HistoSweep** is an optimized, unsupervised, and computationally efficient tool designed to perform a **full quality sweep filtering procedure** for H&E histology images. It identifies **high-quality superpixels** for downstream spatial transcriptomics and image analysis, with major improvements in **speed**, **memory efficiency**, and **platform flexibility**.

---

##  What’s New in HistoSweepO1

-  Optimized for **speed and memory efficiency**
-  Compatible with **Jupyter notebooks**, **Python command-line scripts**, and **Batch job submission**
-  Supports **single-image mode** or **automated multi-image mode**
-  Enables **parallel processing** of multiple H&E images depending on available nodes
-  Supports a wide range of file extensions including whole-slide image formats:  
  `.tiff`, `.svs`, `.ome.tif`, `.tif`, `.jpg`, `.png`, `.ndpi`

---

##  How to Use HistoSweepO1

You can run HistoSweepO1 in one of three ways:

### 1. Interactive Jupyter Notebook (Recommended for First-Time Users)
Run `Run_HistoSweep.ipynb` for an interactive, step-by-step demo of the pipeline.

### 2. Python Command Line Script
Run `run_HistoSweep_single.py` or `run_HistoSweep_multi.py` depending on the mode.

### 3. Batch Job Submission (bsub < run_HistoSweep_job.sh)
Use `run_HistoSweep_job.sh` and select `mode='single'` or `mode='multi'`.

---

## Input Modes

### 🔹 Single-Image Mode

Place your image in a folder under `HE/`, and name it as follows:

| Processing Stage | Filename           | 
|------------------|--------------------|
| Raw (unscaled)   | `he-raw.*`         |
| Scaled           | `he-scaled.*`      |                                    
| Preprocessed     | `he.*`             |                                    


### 🔹 Multi-Image Mode

Place **multiple H&E images** in a folder. The script will:

- **Automatically detect** and process all files with compatible extensions
- **Run each image independently**
- **Enable parallelization** based on available nodes/cores

No specific filename formatting is required.

---

## ⚙️ User-Defined Input Parameters (in notebook or scripts)

### Basic Setup, Preprocessing, and Resolution Settings

```python
# Path prefix to your H&E image folder (inside HistoSweep directory)
HE_prefix = 'HE/demo/'

# Output directory
output_directory = "HistoSweep_Output"

# Flags for preprocessing
need_scaling_flag = False           # True if image resolution ≠ desired pixel_size (e.g. 0.5 µm)
need_preprocessing_flag = False    # True if image dimensions are not divisible by patch_size

# Image resolution parameters
pixel_size_raw = 0.5               # Microns per pixel in original image (provided by scanner metadata)
pixel_size = 0.5                   # Desired resolution (µm/pixel)
patch_size = 16                    # Size of one square patch used throughout processing (16x16 pixels -> ~8µm if pixel_size = 0.5)
```

### Additional Filtering Options

```python
# Artifact removal
density_thresh = 100              # Parameter used to determine the amount of density filtering (e.g., artifact removal)

# Background cleaning for debris
clean_background_flag = True       # Set to False to retain all fibrous/adipose regions
min_size = 10                      # Minimum object size (for debris removal)
