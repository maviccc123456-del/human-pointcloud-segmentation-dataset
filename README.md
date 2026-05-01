# Human Point Cloud Part Segmentation Dataset Pipeline

This repository presents my pipeline for constructing a human body point cloud part-segmentation dataset.

## Overview

This project focuses on preparing human point cloud data for deep learning-based part segmentation.

The pipeline includes:

- Loading registered human mesh data
- Converting mesh vertices into point clouds
- Assigning body-part labels
- Creating train/test splits
- Generating incomplete point clouds to simulate occlusion
- Visualizing ground-truth labels and segmentation results
- Preparing data for PointNet / PointNet++ / DGCNN experiments

## Important Note

Due to the license restrictions of DFAUST/FAUST, the original dataset and the full derived dataset are not included in this repository.

This repository only provides:

- Data preprocessing scripts
- Dataset construction pipeline
- Visualization examples
- Documentation of the labeling process

## Project Motivation

2D image-based pose estimation is mature, but it lacks depth information and may have difficulty understanding spatial distance and occlusion.

Therefore, this project explores 3D point cloud-based human body segmentation. The goal is to study how deep learning models can recognize human body parts from complete and partially missing point clouds.

## Pipeline

1. Load registered human mesh data
2. Extract vertex coordinates as point clouds
3. Assign semantic body-part labels
4. Sample points for model input
5. Create complete and incomplete point cloud versions
6. Train and evaluate segmentation networks
7. Compare robustness under missing-point conditions

## Models

- PointNet
- PointNet++
- DGCNN

## Evaluation Metrics

- Overall Accuracy
- Mean IoU
- Per-part IoU
- Robustness under missing-point conditions

## Japanese Summary

本リポジトリは、人体点群データを用いた部位分割用データセットの作成プロセスをまとめたものです。

人体メッシュデータの読み込み、点群化、部位ラベルの作成、学習・評価データの分割、欠損点群の生成、可視化までを対象としています。

元データセットのライセンス制限により、原始データおよび完全な派生データは公開していません。
