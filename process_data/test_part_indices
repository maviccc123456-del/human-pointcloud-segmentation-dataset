# -*- coding: utf-8 -*-
"""
Visualize and validate body-part vertex indices for DFAUST/FAUST point clouds.

Purpose:
    This script checks whether the vertex-index dictionary can assign body-part
    labels to the correct regions of a registered DFAUST/FAUST human mesh.

目的:
    このスクリプトは、頂点 index の対応表が DFAUST/FAUST の人体点群を
    正しく身体部位ごとに分類できているかを確認するためのものです。

Important:
    The original DFAUST/FAUST HDF5 files are NOT included in this repository.
    Please place them locally and pass the path using --h5-path.

注意:
    DFAUST/FAUST の元 HDF5 ファイルは、このリポジトリには含めません。
    ローカル環境に配置し、--h5-path でパスを指定してください。

Example:
    python process_data/test_part_indices.py \
        --h5-path data/registrations_f.hdf5 \
        --sid 50020 \
        --seq jiggle_on_toes \
        --frame-id 0

Optional:
    python process_data/test_part_indices.py \
        --h5-path data/registrations_f.hdf5 \
        --sid 50020 \
        --seq jiggle_on_toes \
        --frame-id 0 \
        --save-fig figures/part_indices_example.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Import part-index dictionary
# 頂点 index 対応表の読み込み
# =============================================================================
# For public GitHub repositories, use "part_indices_template.py".
# In your local environment, you can replace it with the full "part_indices.py".
#
# 公開 GitHub では "part_indices_template.py" を使用します。
# ローカル環境では、完全版の "part_indices.py" に置き換えて使用できます。
try:
    from part_indices import PART_INDICES  # type: ignore
except ImportError:
    from part_indices_template import PART_INDICES  # type: ignore
    print(
        "[Warning] Full part_indices.py was not found. "
        "Using part_indices_template.py instead."
    )


# =============================================================================
# Visualization settings
# 可視化設定
# =============================================================================
# Colors are only for visualization. They do not affect the dataset labels.
# 色は可視化用です。データセットのラベル値には影響しません。
PART_COLORS = {
    "head": "red",
    "torso": "orange",
    "left_arm": "green",
    "right_arm": "cyan",
    "left_hand": "blue",
    "right_hand": "purple",
    "left_leg": "lime",
    "right_leg": "magenta",
    "left_foot": "brown",
    "right_foot": "pink",
    "unlabeled": "lightgray",
}

DEFAULT_COLOR_CYCLE = [
    "red",
    "orange",
    "green",
    "cyan",
    "blue",
    "purple",
    "lime",
    "magenta",
    "brown",
    "pink",
    "gray",
]


# =============================================================================
# Data loading
# データ読み込み
# =============================================================================
def load_frame_points(
    h5_path: str | Path,
    subject_id: str,
    sequence_name: str,
    frame_id: int,
) -> Tuple[np.ndarray, int]:
    """
    Load one frame from a DFAUST/FAUST HDF5 registration file.

    English:
        DFAUST registration data is usually stored as (6890, 3, F),
        where F is the number of frames. This function converts it to
        (F, 6890, 3) and returns one selected frame.

    日本語:
        DFAUST registration データは通常 (6890, 3, F) の形式です。
        ここで F はフレーム数を表します。この関数では (F, 6890, 3)
        に変換し、指定した 1 フレームを返します.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {h5_path}\n"
            "Please place the DFAUST/FAUST file locally and pass it with --h5-path."
        )

    sequence_key = f"{subject_id}_{sequence_name}"

    with h5py.File(h5_path, "r") as h5_file:
        if sequence_key not in h5_file:
            available_keys = [key for key in h5_file.keys() if key != "faces"]
            preview = available_keys[:10]
            raise KeyError(
                f"Sequence key '{sequence_key}' was not found in {h5_path}.\n"
                f"Available sequence examples: {preview}"
            )

        raw_data = h5_file[sequence_key][:]  # Expected shape: (6890, 3, F)

    if raw_data.ndim != 3 or raw_data.shape[1] != 3:
        raise ValueError(
            f"Unexpected HDF5 data shape: {raw_data.shape}. "
            "Expected something like (6890, 3, F)."
        )

    frames = np.transpose(raw_data, (2, 0, 1))  # (F, 6890, 3)

    if frame_id < 0 or frame_id >= frames.shape[0]:
        raise IndexError(
            f"frame_id={frame_id} is out of range. "
            f"This sequence has {frames.shape[0]} frames."
        )

    return frames[frame_id].astype(np.float32), frames.shape[0]


# =============================================================================
# Index validation
# index 検証
# =============================================================================
def normalize_part_indices(
    part_indices: Dict[str, Iterable[int]]
) -> Dict[str, np.ndarray]:
    """
    Convert all index lists to NumPy arrays.

    English:
        This makes later validation and visualization easier and safer.

    日本語:
        すべての index リストを NumPy 配列に変換し、
        検証と可視化を行いやすくします。
    """
    normalized = {}

    for part_name, indices in part_indices.items():
        array = np.asarray(indices, dtype=np.int64)

        if array.ndim != 1:
            raise ValueError(f"{part_name}: indices must be a 1D array.")

        normalized[part_name] = array

    return normalized


def check_index_range(
    part_indices: Dict[str, np.ndarray],
    num_vertices: int,
) -> None:
    """
    Check whether all indices are within [0, num_vertices - 1].

    English:
        If an index is outside the valid range, the point cloud cannot be
        correctly indexed and the labeling process will fail.

    日本語:
        index が有効範囲外の場合、点群を正しく参照できず、
        ラベル付け処理が失敗します。
    """
    print("========== Index Range Check / index 範囲チェック ==========")

    for part_name, indices in part_indices.items():
        if len(indices) == 0:
            raise ValueError(f"{part_name}: empty index list.")

        min_index = int(indices.min())
        max_index = int(indices.max())

        print(f"{part_name:12s}: count={len(indices):4d}, min={min_index}, max={max_index}")

        if min_index < 0:
            raise ValueError(f"{part_name}: contains negative indices.")

        if max_index >= num_vertices:
            raise ValueError(
                f"{part_name}: index out of range. "
                f"max={max_index}, num_vertices={num_vertices}"
            )

    print("All indices are within the valid range.")
    print("すべての index は有効範囲内です。\n")


def check_overlaps(part_indices: Dict[str, np.ndarray]) -> None:
    """
    Check whether different body parts share the same vertex indices.

    English:
        Overlapped indices mean that one vertex is assigned to multiple body
        parts. In a clean part-segmentation label map, this should usually be
        avoided.

    日本語:
        重複 index がある場合、1 つの頂点が複数の身体部位に割り当てられます。
        部位分割用ラベルとしては、通常この状態は避けるべきです。
    """
    print("========== Overlap Check / 部位重複チェック ==========")

    part_names = list(part_indices.keys())
    has_overlap = False

    for i in range(len(part_names)):
        for j in range(i + 1, len(part_names)):
            name_a = part_names[i]
            name_b = part_names[j]

            set_a = set(part_indices[name_a].tolist())
            set_b = set(part_indices[name_b].tolist())
            overlap = sorted(set_a & set_b)

            if overlap:
                has_overlap = True
                print(f"{name_a} and {name_b}: {len(overlap)} overlapped vertices.")
                print(f"First 20 overlapped indices: {overlap[:20]}")

    if has_overlap:
        raise ValueError(
            "Overlapped indices were found. "
            "Please revise the part-index dictionary before creating labels."
        )

    print("No overlaps were found between body parts.")
    print("身体部位間の重複 index は見つかりませんでした。\n")


def build_vertex_part_map(
    part_indices: Dict[str, np.ndarray],
    num_vertices: int,
) -> np.ndarray:
    """
    Assign a body-part name to each vertex.

    English:
        Vertices that are not covered by the dictionary remain "unlabeled".

    日本語:
        対応表に含まれていない頂点は "unlabeled" として残ります。
    """
    vertex_part = np.full((num_vertices,), "unlabeled", dtype=object)

    for part_name, indices in part_indices.items():
        vertex_part[indices] = part_name

    return vertex_part


def print_coverage_statistics(vertex_part: np.ndarray) -> None:
    """
    Print how many vertices are assigned to each body part.

    English:
        This helps confirm whether the dictionary covers the expected number
        of vertices and whether there are unlabeled points.

    日本語:
        各部位に割り当てられた頂点数を表示し、未ラベル点が残っているかを確認します。
    """
    print("========== Coverage Statistics / ラベル範囲統計 ==========")

    total_vertices = len(vertex_part)
    unique_parts, counts = np.unique(vertex_part, return_counts=True)

    for part_name, count in zip(unique_parts, counts):
        print(f"{part_name:12s}: {int(count):4d} vertices")

    unlabeled_count = int(np.sum(vertex_part == "unlabeled"))
    labeled_count = total_vertices - unlabeled_count

    print("----------------------------------------")
    print(f"Labeled vertices   : {labeled_count}")
    print(f"Unlabeled vertices : {unlabeled_count}")
    print(f"Total vertices     : {total_vertices}")

    if unlabeled_count > 0:
        print(
            "[Warning] Some vertices are still unlabeled. "
            "This may be acceptable for a template file, but should be checked "
            "when using the full dictionary."
        )

    print("==========================================================\n")


# =============================================================================
# Visualization
# 可視化
# =============================================================================
def get_part_color(part_name: str, index: int) -> str:
    """
    Get a visualization color for a body part.

    English:
        If the part name is not in PART_COLORS, use a fallback color cycle.

    日本語:
        PART_COLORS に存在しない部位名の場合、予備の色リストから選択します。
    """
    if part_name in PART_COLORS:
        return PART_COLORS[part_name]

    return DEFAULT_COLOR_CYCLE[index % len(DEFAULT_COLOR_CYCLE)]


def set_axes_equal(ax) -> None:
    """
    Set equal scale for the x, y, and z axes in a 3D plot.

    English:
        Without this, the human body may look unnaturally stretched.

    日本語:
        軸のスケールが異なると人体が不自然に伸びて見えるため、
        x, y, z の表示スケールをそろえます。
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_parts(
    points: np.ndarray,
    vertex_part: np.ndarray,
    subject_id: str,
    sequence_name: str,
    frame_id: int,
    total_frames: int,
    show_unlabeled: bool,
    save_fig: str | Path | None = None,
    show_plot: bool = True,
) -> None:
    """
    Visualize the point cloud with body-part colors.

    English:
        This is a visual sanity check for the part-index dictionary.

    日本語:
        身体部位 index 対応表が正しいかを目視確認するための可視化です。
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    unique_parts = [name for name in np.unique(vertex_part).tolist() if name != "unlabeled"]

    # Draw unlabeled points first so that labeled parts are easier to see.
    # 未ラベル点を先に描画することで、ラベル付き部位を見やすくします。
    if show_unlabeled:
        mask_unlabeled = vertex_part == "unlabeled"
        unlabeled_points = points[mask_unlabeled]

        if len(unlabeled_points) > 0:
            ax.scatter(
                unlabeled_points[:, 0],
                unlabeled_points[:, 1],
                unlabeled_points[:, 2],
                s=3,
                c=PART_COLORS["unlabeled"],
                alpha=0.35,
                label="unlabeled",
            )

    # Draw each body part using its vertex mask.
    # 各身体部位を vertex mask に基づいて描画します。
    for color_index, part_name in enumerate(unique_parts):
        mask = vertex_part == part_name
        part_points = points[mask]

        if len(part_points) == 0:
            continue

        ax.scatter(
            part_points[:, 0],
            part_points[:, 1],
            part_points[:, 2],
            s=6,
            c=get_part_color(part_name, color_index),
            label=part_name,
        )

    ax.set_title(
        f"{subject_id}_{sequence_name} | frame {frame_id}/{total_frames - 1}",
        fontsize=13,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    set_axes_equal(ax)

    legend_elements = []

    if show_unlabeled and np.any(vertex_part == "unlabeled"):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PART_COLORS["unlabeled"],
                markersize=8,
                label="unlabeled",
            )
        )

    for color_index, part_name in enumerate(unique_parts):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=get_part_color(part_name, color_index),
                markersize=8,
                label=part_name,
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="Body Parts / 身体部位",
    )

    plt.tight_layout()

    if save_fig is not None:
        save_path = Path(save_fig)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Command-line interface
# コマンドライン引数
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and visualize a DFAUST/FAUST body-part index dictionary."
        )
    )

    parser.add_argument(
        "--h5-path",
        type=str,
        required=True,
        help="Path to a local DFAUST/FAUST registration HDF5 file.",
    )
    parser.add_argument(
        "--sid",
        type=str,
        required=True,
        help="Subject ID, for example: 50020.",
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="Sequence name, for example: jiggle_on_toes.",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=0,
        help="Frame index to visualize.",
    )
    parser.add_argument(
        "--num-vertices",
        type=int,
        default=6890,
        help="Number of vertices in the registered mesh.",
    )
    parser.add_argument(
        "--hide-unlabeled",
        action="store_true",
        help="Hide unlabeled points in the visualization.",
    )
    parser.add_argument(
        "--save-fig",
        type=str,
        default=None,
        help="Optional path to save the visualization image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save or validate without opening an interactive plot window.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    part_indices = normalize_part_indices(PART_INDICES)

    points, total_frames = load_frame_points(
        h5_path=args.h5_path,
        subject_id=args.sid,
        sequence_name=args.seq,
        frame_id=args.frame_id,
    )

    print(f"Sequence        : {args.sid}_{args.seq}")
    print(f"Total frames    : {total_frames}")
    print(f"Selected frame  : {args.frame_id}")
    print(f"Point shape     : {points.shape}\n")

    check_index_range(part_indices, num_vertices=args.num_vertices)
    check_overlaps(part_indices)

    vertex_part = build_vertex_part_map(
        part_indices=part_indices,
        num_vertices=points.shape[0],
    )
    print_coverage_statistics(vertex_part)

    visualize_parts(
        points=points,
        vertex_part=vertex_part,
        subject_id=args.sid,
        sequence_name=args.seq,
        frame_id=args.frame_id,
        total_frames=total_frames,
        show_unlabeled=not args.hide_unlabeled,
        save_fig=args.save_fig,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
