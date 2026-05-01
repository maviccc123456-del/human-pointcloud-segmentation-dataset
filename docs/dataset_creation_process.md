# Dataset Creation Process

## 1. Data Source

This project uses registered human mesh data as the basis for constructing a human point cloud part-segmentation dataset.

The original data is not redistributed in this repository due to dataset license restrictions.

## 2. Point Cloud Generation

For each mesh frame, the vertex coordinates are extracted as a point cloud.

Each point has 3D coordinates: x, y, z.

The original number of vertices is preserved before model input. Point sampling is performed later in the data loader or model preprocessing stage.

## 3. Body Part Labeling

Body part labels are manually or semi-automatically assigned based on mesh regions.

Example body parts include:

- head
- torso
- left arm
- right arm
- left hand
- right hand
- left leg
- right leg
- left foot
- right foot

## 4. Train/Test Split

The processed point clouds are divided into training and testing sets.

The test set is not used during training. It is only used for evaluating model performance after training.

## 5. Missing Point Cloud Generation

To simulate real-world scanning conditions, local regions of the human body point cloud are removed.

This allows comparison between:

- complete point clouds
- incomplete point clouds

The purpose is to evaluate how robust each model is when part of the human body point cloud is missing.

## 6. Target Models

The prepared dataset can be used for experiments with point cloud segmentation networks such as:

- PointNet
- PointNet++
- DGCNN

## 7. Evaluation

The segmentation performance can be evaluated using:

- Overall Accuracy
- Mean IoU
- Per-part IoU

## 8. Purpose

The final goal is to study human point cloud part segmentation under realistic incomplete-data conditions.
