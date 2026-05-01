# -*- coding: utf-8 -*-
"""
part_indices_template.py

Public template for body-part vertex indices.
人体部位ごとの頂点 index を定義するための公開用テンプレートです。

IMPORTANT / 重要:
- This public file does NOT include the full DFAUST/FAUST vertex-index mapping.
- この公開用ファイルには、DFAUST/FAUST の完全な頂点 index 対応表は含めていません。

Why / 理由:
- The complete index lists are tied to a licensed human mesh topology.
- 完全な index リストは、ライセンス付き人体メッシュの topology に依存しているためです。

How to use locally / ローカルでの使い方:
1. Copy this file as `part_indices.py` in your local environment.
   このファイルをローカル環境で `part_indices.py` としてコピーします。
2. Replace the example index arrays with your private full index arrays.
   下のサンプル index 配列を、自分の完全な index 配列に置き換えます。
3. Keep the complete `part_indices.py` out of public GitHub repositories.
   完全版の `part_indices.py` は公開 GitHub リポジトリに置かないでください。
"""

from __future__ import annotations

import numpy as np


# Number of vertices in the registered DFAUST/FAUST-style mesh.
# DFAUST/FAUST 形式の registration mesh における頂点数です。
NUM_VERTICES = 6890


# Semantic label IDs used for human body part segmentation.
# 人体部位分割で使用する semantic label ID です。
#
# Keep this order consistent with the dataset-generation script.
# この順番は、データセット生成スクリプトと一致させてください。
LABEL_MAP = {
    "head": 0,
    "torso": 1,
    "left_arm": 2,
    "right_arm": 3,
    "left_hand": 4,
    "right_hand": 5,
    "left_foot": 6,
    "right_foot": 7,
}


# -----------------------------------------------------------------------------
# Public example indices
# 公開用サンプル index
# -----------------------------------------------------------------------------
# These are NOT the real full labels.
# これは実際の完全なラベルではありません。
#
# They are only small placeholders to show the expected file format.
# ファイル形式を示すための小さな placeholder です。
#
# For real experiments, replace them with your private full index lists.
# 実験で使う場合は、自分の完全な index リストに置き換えてください。

head_idx = np.array([0, 1, 2], dtype=np.int64)
torso_idx = np.array([3, 4, 5], dtype=np.int64)
left_arm_idx = np.array([6, 7, 8], dtype=np.int64)
right_arm_idx = np.array([9, 10, 11], dtype=np.int64)
left_hand_idx = np.array([12, 13, 14], dtype=np.int64)
right_hand_idx = np.array([15, 16, 17], dtype=np.int64)
left_foot_idx = np.array([18, 19, 20], dtype=np.int64)
right_foot_idx = np.array([21, 22, 23], dtype=np.int64)


# Dictionary used by the preprocessing script.
# 前処理スクリプトから参照される辞書です。
PART_INDICES = {
    "head": head_idx,
    "torso": torso_idx,
    "left_arm": left_arm_idx,
    "right_arm": right_arm_idx,
    "left_hand": left_hand_idx,
    "right_hand": right_hand_idx,
    "left_foot": left_foot_idx,
    "right_foot": right_foot_idx,
}


def summarize_part_indices() -> None:
    """
    Print the number of vertices assigned to each body part.
    各身体部位に割り当てられた頂点数を表示します。

    This function is useful for checking whether the label definition has been
    loaded correctly before running the dataset-generation script.
    データセット生成前に、ラベル定義が正しく読み込まれているか確認するために使います。
    """
    print("========== Body-part index summary / 部位 index の概要 ==========")
    total = 0
    for part_name, indices in PART_INDICES.items():
        label = LABEL_MAP[part_name]
        count = len(indices)
        total += count
        print(f"{part_name:12s}: {count:4d} vertices, label={label}")
    print("---------------------------------------------------------------")
    print(f"Total labeled vertices / ラベル付き頂点数: {total}")
    print(f"Expected full mesh vertices / 想定される全頂点数: {NUM_VERTICES}")
    print("Note: this public template intentionally contains only example indices.")
    print("注意: この公開用テンプレートにはサンプル index のみが含まれています。")


def validate_template_indices() -> None:
    """
    Validate the placeholder index arrays.
    placeholder の index 配列を検証します。

    This only checks basic problems such as negative indices, out-of-range
    indices, and duplicated indices across body parts.
    負の index、範囲外 index、部位間の重複 index などの基本的な問題だけを確認します。
    """
    used: set[int] = set()

    for part_name, indices in PART_INDICES.items():
        if not isinstance(indices, np.ndarray):
            raise TypeError(f"{part_name} must be a numpy.ndarray")

        if indices.dtype != np.int64:
            raise TypeError(f"{part_name} must use dtype=np.int64")

        if len(indices) == 0:
            raise ValueError(f"{part_name} has no indices / index が空です")

        if np.any(indices < 0):
            raise ValueError(f"{part_name} contains negative indices / 負の index があります")

        if np.any(indices >= NUM_VERTICES):
            raise ValueError(f"{part_name} contains out-of-range indices / 範囲外 index があります")

        overlap = used.intersection(indices.tolist())
        if overlap:
            raise ValueError(
                f"{part_name} overlaps with previous parts: {sorted(overlap)[:20]} / "
                f"前の部位と index が重複しています"
            )

        used.update(indices.tolist())

    print("Template indices are valid. / テンプレート index の検証が完了しました。")


if __name__ == "__main__":
    summarize_part_indices()
    validate_template_indices()
