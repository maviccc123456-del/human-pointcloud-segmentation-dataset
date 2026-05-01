# Human Point Cloud Part Segmentation Dataset Pipeline

This repository presents a data preparation pipeline for human point cloud part segmentation based on DFAUST/FAUST-style registered human mesh data.

The main goal of this project is to construct vertex-level body-part labels for human point clouds and prepare the data for deep learning-based point cloud segmentation models such as PointNet, PointNet++, and DGCNN.

---

## Overview

This project focuses on the following tasks:

- Reading DFAUST/FAUST HDF5 registration files
- Checking the internal structure of the HDF5 dataset
- Defining body-part labels using vertex index mappings
- Visualizing whether each point is correctly assigned to a human body part
- Generating point cloud segmentation samples
- Preparing data for point cloud segmentation models

The generated data format is designed for semantic part segmentation:

```text
points: (6890, 3)
labels: (6890,)
```

---

## Important Note

Due to the license restrictions of DFAUST/FAUST, the original HDF5 files and the generated full segmentation dataset are not included in this public repository.

This repository only provides:

- Public preprocessing scripts
- A body-part index template or public version
- Validation scripts
- Visualization examples
- Documentation of the dataset construction process

The original data files should be prepared locally and should not be uploaded to GitHub.

---

## Repository Structure

```text
human-pointcloud-segmentation-dataset/
├── README.md
├── README_en.md
├── README_ja.md
├── .gitignore
├── data/
│   └── .gitkeep
├── process_data/
│   ├── make_dfaust_part_dataset_public
│   ├── part_indices
│   ├── test_data_set
│   └── test_part_indices
└── figures/
    ├── 1.png
    ├── 2.png
    ├── 3.png
    └── 4.png
```

---

## Data Directory

The `data/` directory is used only in the local environment.

Expected local structure:

```text
data/
├── registrations_f.hdf5
├── registrations_m.hdf5
└── DFAUST_PART_SEG_ALL/
```

Explanation:

- `registrations_f.hdf5`: registered female human mesh data
- `registrations_m.hdf5`: registered male human mesh data
- `DFAUST_PART_SEG_ALL/`: generated part-segmentation dataset

These files are not included in this repository.

---

## Body-Part Label Definition

The body-part labels are defined using a vertex index dictionary.

Each vertex index is assigned to one semantic body part, such as:

- head
- torso
- left arm
- right arm
- left hand
- right hand
- left foot
- right foot

Once the vertex-level body-part dictionary is defined, the generated labels can be used as common ground-truth annotations for different point cloud segmentation models.

---

## Visualization of Body-Part Label Mapping

The following figures show visualization results of the manually defined body-part index dictionary.

These images are not model prediction results.  
They are used to verify whether each vertex index is correctly assigned to the corresponding human body part.

### Example 1

![Part Label Example 1](figures/1.png)

### Example 2

![Part Label Example 2](figures/2.png)

### Example 3

![Part Label Example 3](figures/3.png)

### Example 4

![Part Label Example 4](figures/4.png)

These visualizations confirm that the manually defined vertex-index dictionary can assign each point to a semantic human body part.

---

## Processing Pipeline

The overall processing pipeline is as follows:

1. Load DFAUST/FAUST HDF5 registration data
2. Check the internal dataset structure
3. Define body-part index mappings
4. Validate whether body-part indices overlap
5. Assign semantic labels to each vertex
6. Visualize the labeled point cloud
7. Generate segmentation samples
8. Save the processed dataset locally

---

## Scripts

### 1. View HDF5 Structure

`test_data_set` is used to check the internal structure of the HDF5 file.

Example:

```bash
python process_data/test_data_set --h5-path data/registrations_f.hdf5
```

This script prints dataset names, shapes, and data types inside the HDF5 file.

---

### 2. Validate Body-Part Index Dictionary

`test_part_indices` is used to visualize whether the body-part index dictionary correctly separates the human point cloud into different body regions.

Example:

```bash
python process_data/test_part_indices \
  --h5-path data/registrations_f.hdf5 \
  --sid 50020 \
  --seq jiggle_on_toes \
  --frame-id 0
```

This script is mainly used for checking:

- whether each body part is correctly assigned
- whether different body parts overlap
- whether unlabeled points exist
- whether the visualization looks reasonable

---

### 3. Generate Part-Segmentation Dataset

`make_dfaust_part_dataset_public` is used to generate local point cloud part-segmentation samples.

Example:

```bash
python process_data/make_dfaust_part_dataset_public \
  --male-h5 data/registrations_m.hdf5 \
  --female-h5 data/registrations_f.hdf5 \
  --save-root data/DFAUST_PART_SEG_ALL
```

The generated dataset will be saved locally under:

```text
data/DFAUST_PART_SEG_ALL/
```

The generated full dataset is not uploaded to GitHub.

---

## Target Models

The generated labels can be used for training and evaluating point cloud segmentation networks such as:

- PointNet
- PointNet++
- DGCNN

This project focuses on dataset preparation and label validation before model training.

---

## Evaluation Plan

After preparing the dataset, segmentation performance can be evaluated using:

- Overall Accuracy
- Mean IoU
- Per-part IoU
- Robustness under incomplete point cloud conditions

---

## License Notice

The original DFAUST/FAUST dataset is subject to its own license terms.

This repository does not redistribute the original dataset or the generated full dataset.  
Only scripts, documentation, and visualization examples are provided.
