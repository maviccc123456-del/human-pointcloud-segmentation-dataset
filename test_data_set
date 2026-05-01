# -*- coding: utf-8 -*-
"""
View the internal structure of a DFAUST/FAUST HDF5 file.

English:
This script prints the group/dataset structure of an HDF5 file.
It is useful for checking sequence names, array shapes, and data types
before building a point-cloud segmentation dataset.

日本語:
このスクリプトは HDF5 ファイル内部のグループ・データセット構造を表示します。
点群分割データセットを作成する前に、シーケンス名、配列 shape、データ型を確認するために使用します。

Example / 実行例:
    python process_data/view_data.py --h5-path data/registrations_f.hdf5

Optional / 任意:
    python process_data/view_data.py --h5-path data/registrations_f.hdf5 --show-attrs
    python process_data/view_data.py --h5-path data/registrations_f.hdf5 --output hdf5_structure.txt
"""

import argparse
from pathlib import Path

import h5py


def print_hdf5_structure(name, obj, show_attrs=False, lines=None):
    """
    Print one item in the HDF5 tree.

    English:
    If the item is a dataset, print its path, shape, and dtype.
    If the item is a group, print only its path.

    日本語:
    対象がデータセットの場合は、パス・shape・dtype を表示します。
    グループの場合は、パスのみを表示します。
    """
    if isinstance(obj, h5py.Dataset):
        message = f"[Dataset] {name} | shape={obj.shape} | dtype={obj.dtype}"
    else:
        message = f"[Group]   {name}"

    print(message)

    if lines is not None:
        lines.append(message)

    # English: Optionally print HDF5 attributes.
    # 日本語: 必要に応じて HDF5 の属性情報も表示します。
    if show_attrs and len(obj.attrs) > 0:
        for key, value in obj.attrs.items():
            attr_message = f"    attr: {key} = {value}"
            print(attr_message)
            if lines is not None:
                lines.append(attr_message)


def summarize_hdf5_file(h5_path, show_attrs=False, output_path=None):
    """
    Open an HDF5 file and print its internal structure.

    English:
    This function does not modify the dataset. It only reads metadata
    such as group names, dataset shapes, and data types.

    日本語:
    この関数はデータセットを変更しません。
    グループ名、データセット shape、データ型などのメタ情報だけを読み取ります。
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    lines = []
    header = f"========== HDF5 Structure: {h5_path} =========="
    print(header)
    lines.append(header)

    with h5py.File(h5_path, "r") as f:
        # English: Print root-level keys first for a quick overview.
        # 日本語: まず root 階層の key を表示し、全体像を確認します。
        root_keys_message = f"Root keys: {list(f.keys())}"
        print(root_keys_message)
        print("-----------------------------------------------")
        lines.append(root_keys_message)
        lines.append("-----------------------------------------------")

        f.visititems(
            lambda name, obj: print_hdf5_structure(
                name=name,
                obj=obj,
                show_attrs=show_attrs,
                lines=lines,
            )
        )

    footer = "==================== Finished ===================="
    print(footer)
    lines.append(footer)

    # English: Save the printed structure to a text file if requested.
    # 日本語: 必要に応じて、表示結果を txt ファイルとして保存します。
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved structure report to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="View the internal structure of a DFAUST/FAUST HDF5 file."
    )

    parser.add_argument(
        "--h5-path",
        type=str,
        required=True,
        help="Path to the local HDF5 file, e.g., data/registrations_f.hdf5",
    )

    parser.add_argument(
        "--show-attrs",
        action="store_true",
        help="Print HDF5 attributes if they exist.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the structure report as a text file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    summarize_hdf5_file(
        h5_path=args.h5_path,
        show_attrs=args.show_attrs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
