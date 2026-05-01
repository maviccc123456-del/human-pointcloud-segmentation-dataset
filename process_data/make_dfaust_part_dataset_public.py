# -*- coding: utf-8 -*-
"""
Public dataset generation script for DFAUST/FAUST-based human point cloud part segmentation.

DFAUST/FAUST に基づく人体点群部位分割データセット作成用の公開版スクリプトです。

Important / 注意:
    - The original DFAUST/FAUST data is NOT included in this repository.
      元の DFAUST/FAUST データはこのリポジトリには含めません。
    - This script only shows the preprocessing pipeline.
      本スクリプトは前処理パイプラインを示すためのものです。
    - Do not upload generated .npz files if they are derived from licensed datasets.
      ライセンス付きデータから生成した .npz ファイルは公開しないでください。

Expected HDF5 format / 想定する HDF5 形式:
    Each motion sequence is stored as:
        (6890, 3, F)

    where:
        6890: number of mesh vertices / メッシュ頂点数
        3   : x, y, z coordinates / x, y, z 座標
        F   : number of frames / フレーム数

Output format / 出力形式:
    Each frame is saved as one compressed .npz file:
        points: (6890, 3)
        labels: (6890,)

Usage example / 実行例:
    python make_dfaust_part_dataset.py \
        --male-h5 path/to/registrations_m.hdf5 \
        --female-h5 path/to/registrations_f.hdf5 \
        --save-root path/to/output_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import h5py
import numpy as np

try:
    from part_indices import PART_INDICES, LABEL_MAP
except ImportError as exc:
    raise ImportError(
        "Could not import PART_INDICES and LABEL_MAP from part_indices.py.\n"
        "Please prepare part_indices.py locally.\n\n"
        "part_indices.py から PART_INDICES と LABEL_MAP を読み込めませんでした。\n"
        "ローカル環境で part_indices.py を用意してください。"
    ) from exc


# Number of vertices in the registered DFAUST/FAUST mesh.
# DFAUST/FAUST registration mesh の頂点数。
NUM_VERTICES = 6890


# Default test subjects.
# デフォルトのテスト用 subject ID。
DEFAULT_TEST_SUBJECTS = {
    "50027",  # male subject / 男性 subject
    "50025",  # female subject / 女性 subject
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    コマンドライン引数を読み込みます。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a human point cloud part-segmentation dataset from local "
            "DFAUST/FAUST registration files."
        )
    )

    parser.add_argument(
        "--male-h5",
        type=str,
        default=None,
        help="Path to local male registration HDF5 file. / 男性 registration HDF5 のローカルパス。",
    )
    parser.add_argument(
        "--female-h5",
        type=str,
        default=None,
        help="Path to local female registration HDF5 file. / 女性 registration HDF5 のローカルパス。",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        required=True,
        help="Output directory for generated samples. / 生成サンプルの保存先。",
    )
    parser.add_argument(
        "--test-subjects",
        type=str,
        nargs="*",
        default=sorted(DEFAULT_TEST_SUBJECTS),
        help=(
            "Subject IDs used for the test split. "
            "/ テストデータとして使用する subject ID。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check dataset structure without saving .npz files. "
            "/ .npz を保存せず、データ構造だけ確認します。"
        ),
    )

    return parser.parse_args()


def build_h5_file_dict(args: argparse.Namespace) -> Dict[str, Path]:
    """
    Build a dictionary of available HDF5 files.

    指定された HDF5 ファイルだけを辞書にまとめます。

    Note / 注意:
        Hard-coded absolute paths should not be used in public repositories.
        公開リポジトリでは絶対パスを直接書かないようにします。
    """
    h5_files: Dict[str, Path] = {}

    if args.male_h5 is not None:
        h5_files["male"] = Path(args.male_h5)

    if args.female_h5 is not None:
        h5_files["female"] = Path(args.female_h5)

    if not h5_files:
        raise ValueError(
            "At least one of --male-h5 or --female-h5 must be provided.\n"
            "--male-h5 または --female-h5 の少なくとも一方を指定してください。"
        )

    for gender, path in h5_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{gender} HDF5 file was not found: {path}\n"
                f"{gender} の HDF5 ファイルが見つかりません: {path}"
            )

    return h5_files


def convert_to_frames(data: np.ndarray) -> np.ndarray:
    """
    Convert DFAUST/FAUST sequence data into frame-first format.

    DFAUST/FAUST の sequence データを frame-first 形式に変換します。

    Input / 入力:
        (6890, 3, F)

    Output / 出力:
        (F, 6890, 3)
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, but got shape: {data.shape}")

    if data.shape[0] == NUM_VERTICES and data.shape[1] == 3:
        return np.transpose(data, (2, 0, 1))

    raise ValueError(
        f"Unsupported data shape: {data.shape}. "
        f"Expected ({NUM_VERTICES}, 3, F)."
    )


def get_subject_id(seq_name: str) -> str:
    """
    Extract subject ID from a sequence name.

    sequence 名から subject ID を取り出します。

    Example / 例:
        50002_chicken_wings -> 50002
        50004_jumping_jacks -> 50004
    """
    return seq_name.split("_")[0]


def get_split(seq_name: str, test_subjects: Set[str]) -> str:
    """
    Decide whether the sequence belongs to train or test split.

    sequence が train/test のどちらに属するかを決定します。

    Important / 重要:
        Splitting by subject helps avoid training and testing on frames
        from the same person.

        subject 単位で分割することで、同一人物のフレームが
        train と test の両方に入ることを避けます。
    """
    subject_id = get_subject_id(seq_name)
    return "test" if subject_id in test_subjects else "train"


def validate_part_indices(num_vertices: int = NUM_VERTICES) -> None:
    """
    Validate body-part vertex indices.

    身体部位ごとの頂点 index が正しいか確認します。

    Checks / 確認内容:
        - each part has at least one vertex / 各部位が空ではないこと
        - no negative index / 負の index がないこと
        - no out-of-range index / 範囲外 index がないこと
        - total number of labeled vertices equals 6890 / 合計が 6890 であること
    """
    print("========== Validate body-part indices / 部位 index の検証 ==========")

    total_points = 0

    for part_name, idx in PART_INDICES.items():
        idx = np.asarray(idx)

        if len(idx) == 0:
            raise ValueError(f"{part_name} has no vertex index. / index が空です。")

        if idx.min() < 0:
            raise ValueError(f"{part_name} has a negative index. / 負の index があります。")

        if idx.max() >= num_vertices:
            raise ValueError(
                f"{part_name} has an out-of-range index: "
                f"max={idx.max()}, num_vertices={num_vertices}"
            )

        total_points += len(idx)
        print(f"{part_name:16s}: {len(idx):4d} points, label={LABEL_MAP[part_name]}")

    print("------------------------------------------------------------------")
    print("Total labeled vertices / 標注済み頂点数:", total_points)
    print("Expected vertices / 想定頂点数:", num_vertices)

    if total_points != num_vertices:
        raise ValueError(
            f"The total number of labeled vertices must be {num_vertices}, "
            f"but got {total_points}."
        )

    print("Index validation finished. / index 検証完了。\n")


def check_overlaps() -> None:
    """
    Check whether different body parts share the same vertex index.

    異なる身体部位の間で同じ頂点 index が重複していないか確認します。
    """
    print("========== Check overlaps / 部位間の重複確認 ==========")

    part_names = list(PART_INDICES.keys())
    has_overlap = False

    for i in range(len(part_names)):
        for j in range(i + 1, len(part_names)):
            name_a = part_names[i]
            name_b = part_names[j]

            set_a = set(np.asarray(PART_INDICES[name_a]).tolist())
            set_b = set(np.asarray(PART_INDICES[name_b]).tolist())

            overlap = set_a & set_b

            if overlap:
                has_overlap = True
                print(
                    f"{name_a} and {name_b} share {len(overlap)} vertices. "
                    f"/ {name_a} と {name_b} に重複 index があります。"
                )
                print("First 20 overlaps / 最初の20個:", sorted(list(overlap))[:20])
                print()

    if has_overlap:
        raise ValueError(
            "Overlapping body-part labels were found. "
            "Please fix part_indices.py first.\n"
            "部位ラベルに重複があります。先に part_indices.py を修正してください。"
        )

    print("No overlaps found. / 重複は見つかりませんでした。")
    print("=====================================================\n")


def make_seg_labels(num_vertices: int = NUM_VERTICES) -> np.ndarray:
    """
    Create one semantic label for each mesh vertex.

    各メッシュ頂点に対して1つの意味ラベルを作成します。

    Output / 出力:
        labels.shape = (6890,)

    Note / 注意:
        Since all registered frames share the same topology, the same label
        array can be used for every frame.

        registration 済みデータは同じ topology を共有するため、
        すべての frame に同じ label 配列を使用できます。
    """
    # Temporary value for detecting unlabeled vertices.
    # 未標注点を検出するための一時値。
    labels = np.full((num_vertices,), -999, dtype=np.int64)

    for part_name, idx in PART_INDICES.items():
        label = LABEL_MAP[part_name]
        labels[np.asarray(idx)] = label

    unlabeled_mask = labels == -999
    unlabeled_count = int(np.sum(unlabeled_mask))

    if unlabeled_count > 0:
        unlabeled_idx = np.where(unlabeled_mask)[0]
        raise ValueError(
            f"{unlabeled_count} vertices are still unlabeled. "
            f"First 20: {unlabeled_idx[:20]}"
        )

    return labels


def print_label_statistics(seg_labels: np.ndarray) -> None:
    """
    Print the number of vertices in each body-part class.

    各身体部位クラスに含まれる頂点数を表示します。
    """
    print("========== Label statistics / ラベル統計 ==========")

    total_count = 0

    for part_name, label in LABEL_MAP.items():
        count = int(np.sum(seg_labels == label))
        total_count += count
        print(f"{part_name:16s}: {count:4d} points, label={label}")

    print("---------------------------------------------------")
    print("Total labeled points / 標注済み点数:", total_count)
    print("Total points / 全点数:", len(seg_labels))

    if total_count != len(seg_labels):
        raise ValueError(
            f"Label count mismatch: labeled={total_count}, total={len(seg_labels)}"
        )

    print("All vertices are labeled. / すべての頂点が標注されています。")
    print("===================================================\n")


def save_label_map(save_root: Path) -> None:
    """
    Save label ID and body-part name mapping.

    label ID と身体部位名の対応関係を保存します。
    """
    save_root.mkdir(parents=True, exist_ok=True)

    label_path = save_root / "label_map.txt"

    with label_path.open("w", encoding="utf-8") as f:
        for part_name, label in LABEL_MAP.items():
            f.write(f"{label} {part_name}\n")

    print("Saved label map / label_map を保存:", label_path)


def save_dataset_info(save_root: Path, test_subjects: Iterable[str]) -> None:
    """
    Save public dataset information.

    公開可能なデータセット情報だけを保存します。

    Important / 重要:
        This function does NOT save local absolute paths.
        ローカルの絶対パスは保存しません。
    """
    save_root.mkdir(parents=True, exist_ok=True)

    info_path = save_root / "dataset_info.txt"

    with info_path.open("w", encoding="utf-8") as f:
        f.write("DFAUST/FAUST Human Part Segmentation Dataset Pipeline\n")
        f.write("====================================================\n\n")

        f.write("Important note:\n")
        f.write("- Original DFAUST/FAUST files are not redistributed.\n")
        f.write("- Generated full datasets should not be uploaded publicly if they are derived from licensed data.\n\n")

        f.write("Data format:\n")
        f.write("Each generated .npz contains:\n")
        f.write("points: (6890, 3)\n")
        f.write("labels: (6890,)\n\n")

        f.write("Test subjects:\n")
        for sid in sorted(test_subjects):
            f.write(f"{sid}\n")

        f.write("\nLabel map:\n")
        for part_name, label in LABEL_MAP.items():
            f.write(f"{label} {part_name}\n")

    print("Saved dataset info / dataset_info を保存:", info_path)


def save_sample(
    points: np.ndarray,
    labels: np.ndarray,
    save_root: Path,
    split: str,
    seq_name: str,
    frame_id: int,
    gender: str,
) -> None:
    """
    Save one complete human point cloud segmentation sample.

    1フレーム分の人体点群分割サンプルを保存します。

    Each sample / 各サンプル:
        points: (6890, 3)
        labels: (6890,)
    """
    save_dir = save_root / split
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_name = f"{gender}_{seq_name}_frame{frame_id:05d}.npz"
    save_path = save_dir / sample_name

    np.savez_compressed(
        save_path,
        points=points.astype(np.float32),
        labels=labels.astype(np.int64),
        seq_name=seq_name,
        frame_id=np.array(frame_id, dtype=np.int64),
        subject_id=get_subject_id(seq_name),
        gender=gender,
    )


def count_sequences_and_frames(h5_files: Dict[str, Path]) -> None:
    """
    Count the number of motion sequences and frames before processing.

    処理前に motion sequence 数と frame 数を確認します。
    """
    print("========== Dataset statistics / データ統計 ==========")

    total_sequences = 0
    total_frames = 0

    for gender, h5_path in h5_files.items():
        gender_sequences = 0
        gender_frames = 0

        with h5py.File(h5_path, "r") as f:
            seq_names = [name for name in f.keys() if name != "faces"]

            for seq_name in seq_names:
                data = f[seq_name]

                if data.shape[0] != NUM_VERTICES or data.shape[1] != 3:
                    raise ValueError(f"{seq_name} has unexpected shape: {data.shape}")

                frames = data.shape[2]
                gender_sequences += 1
                gender_frames += frames

        total_sequences += gender_sequences
        total_frames += gender_frames

        print(f"{gender:6s}: sequences={gender_sequences}, frames={gender_frames}")

    print("----------------------------------------------------")
    print(f"total : sequences={total_sequences}, frames={total_frames}")
    print(f"Estimated samples / 予想サンプル数: {total_frames}")
    print("====================================================\n")


def process_h5_files(
    h5_files: Dict[str, Path],
    save_root: Path,
    seg_labels: np.ndarray,
    test_subjects: Set[str],
    dry_run: bool = False,
) -> None:
    """
    Process all HDF5 files and save frame-level point cloud samples.

    すべての HDF5 ファイルを処理し、frame 単位の点群サンプルとして保存します。
    """
    train_count = 0
    test_count = 0

    train_subjects: Set[str] = set()
    test_subjects_found: Set[str] = set()

    for gender, h5_path in h5_files.items():
        print(f"\n========== Processing {gender} data / {gender} データ処理 ==========")
        print("HDF5 file name / HDF5 ファイル名:", h5_path.name)

        with h5py.File(h5_path, "r") as f:
            seq_names = [name for name in f.keys() if name != "faces"]

            print(f"Number of sequences / sequence 数: {len(seq_names)}")
            print("First sequences / 最初の sequence:", seq_names[:5])
            print()

            for seq_id, seq_name in enumerate(seq_names):
                # Load one motion sequence.
                # 1つの motion sequence を読み込みます。
                raw_data = f[seq_name][:]              # (6890, 3, F)
                frames = convert_to_frames(raw_data)   # (F, 6890, 3)

                split = get_split(seq_name, test_subjects)
                subject_id = get_subject_id(seq_name)
                num_frames = frames.shape[0]

                if split == "train":
                    train_subjects.add(subject_id)
                else:
                    test_subjects_found.add(subject_id)

                print(
                    f"[{gender} {seq_id + 1}/{len(seq_names)}] "
                    f"{seq_name}, frames={num_frames}, split={split}"
                )

                for frame_id in range(num_frames):
                    points = frames[frame_id]  # (6890, 3)

                    if not dry_run:
                        save_sample(
                            points=points,
                            labels=seg_labels,
                            save_root=save_root,
                            split=split,
                            seq_name=seq_name,
                            frame_id=frame_id,
                            gender=gender,
                        )

                    if split == "train":
                        train_count += 1
                    else:
                        test_count += 1

    print("\n========== Finished / 作成完了 ==========")
    print("Output directory / 保存先:", save_root)
    print("train samples:", train_count)
    print("test samples :", test_count)
    print("train subjects:", sorted(train_subjects))
    print("test subjects :", sorted(test_subjects_found))
    print("dry run:", dry_run)
    print("========================================")


def main() -> None:
    """
    Main preprocessing pipeline.

    メインの前処理パイプラインです。
    """
    args = parse_args()

    save_root = Path(args.save_root)
    test_subjects = set(args.test_subjects)
    h5_files = build_h5_file_dict(args)

    # Step 1: validate the manually defined body-part labels.
    # Step 1: 手動で定義した身体部位ラベルを検証します。
    validate_part_indices(num_vertices=NUM_VERTICES)
    check_overlaps()

    # Step 2: inspect data size before saving.
    # Step 2: 保存前にデータ規模を確認します。
    count_sequences_and_frames(h5_files)

    # Step 3: create labels shared by all registered frames.
    # Step 3: registration 済み全 frame で共通する label を作成します。
    seg_labels = make_seg_labels(num_vertices=NUM_VERTICES)
    print_label_statistics(seg_labels)

    # Step 4: save public metadata only.
    # Step 4: 公開可能なメタ情報だけを保存します。
    if not args.dry_run:
        save_label_map(save_root)
        save_dataset_info(save_root, test_subjects)

    # Step 5: generate frame-level samples.
    # Step 5: frame 単位のサンプルを生成します。
    process_h5_files(
        h5_files=h5_files,
        save_root=save_root,
        seg_labels=seg_labels,
        test_subjects=test_subjects,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
