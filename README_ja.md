# 人体点群部位分割データセット作成パイプライン

本リポジトリは、DFAUST/FAUST 形式の registered human mesh data を用いて、人体点群の部位分割用データセットを作成するための処理パイプラインをまとめたものです。

主な目的は、人体点群に対して頂点レベルの身体部位ラベルを作成し、PointNet、PointNet++、DGCNN などの深層学習ベースの点群セグメンテーションモデルに利用できる形式へ整備することです。

---

## 概要

本プロジェクトでは、以下の処理を行います。

- DFAUST/FAUST の HDF5 registration file の読み込み
- HDF5 データセット内部構造の確認
- 頂点 index に基づく身体部位ラベルの定義
- 各点が正しい身体部位に割り当てられているかの可視化確認
- 点群セグメンテーション用サンプルの生成
- 点群セグメンテーションモデル用データの準備

生成されるデータ形式は以下の通りです。

```text
points: (6890, 3)
labels: (6890,)
```

---

## 重要な注意事項

DFAUST/FAUST のライセンス制限により、元の HDF5 ファイルおよび生成済みの完全な分割データセットは、この公開リポジトリには含めていません。

本リポジトリで公開しているものは以下のみです。

- 公開用の前処理スクリプト
- 身体部位 index のテンプレートまたは公開用バージョン
- 検証用スクリプト
- 可視化例
- データセット作成プロセスの説明

元データファイルはローカル環境で準備し、GitHub にはアップロードしません。

---

## リポジトリ構成

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

## data ディレクトリについて

`data/` ディレクトリはローカル環境でのみ使用します。

想定されるローカル構成は以下の通りです。

```text
data/
├── registrations_f.hdf5
├── registrations_m.hdf5
└── DFAUST_PART_SEG_ALL/
```

各ファイル・ディレクトリの意味は以下の通りです。

- `registrations_f.hdf5`: 女性人体の registered mesh data
- `registrations_m.hdf5`: 男性人体の registered mesh data
- `DFAUST_PART_SEG_ALL/`: 生成済みの部位分割用データセット

これらのファイルは本リポジトリには含めていません。

---

## 身体部位ラベルの定義

身体部位ラベルは、頂点 index 辞書を用いて定義しています。

各頂点 index は、以下のような意味的な身体部位に割り当てられます。

- head
- torso
- left arm
- right arm
- left hand
- right hand
- left foot
- right foot

頂点レベルの身体部位 index 辞書を定義することで、生成されたラベルは PointNet、PointNet++、DGCNN などの異なる点群セグメンテーションモデルに共通の ground-truth annotation として利用できます。

---

## 身体部位ラベルの可視化

以下の図は、手動で定義した身体部位 index 辞書を用いた可視化結果です。

これらの画像はモデルの推論結果ではありません。  
各頂点 index が対応する身体部位に正しく割り当てられているかを確認するためのものです。

### Example 1

![Part Label Example 1](figures/1.png)

### Example 2

![Part Label Example 2](figures/2.png)

### Example 3

![Part Label Example 3](figures/3.png)

### Example 4

![Part Label Example 4](figures/4.png)

これらの可視化結果により、手動で定義した頂点 index 辞書が各点を意味的な身体部位に割り当てられることを確認できます。

---

## 処理パイプライン

全体の処理手順は以下の通りです。

1. DFAUST/FAUST の HDF5 registration data を読み込む
2. データセット内部構造を確認する
3. 身体部位 index mapping を定義する
4. 身体部位 index に重複がないか確認する
5. 各頂点に意味的ラベルを割り当てる
6. ラベル付き点群を可視化する
7. セグメンテーション用サンプルを生成する
8. 処理済みデータセットをローカルに保存する

---

## スクリプト

### 1. HDF5 構造の確認

`test_data_set` は、HDF5 ファイル内部構造を確認するためのスクリプトです。

実行例：

```bash
python process_data/test_data_set --h5-path data/registrations_f.hdf5
```

このスクリプトでは、HDF5 ファイル内のデータセット名、shape、データ型を表示します。

---

### 2. 身体部位 index 辞書の検証

`test_part_indices` は、身体部位 index 辞書が人体点群を正しい部位に分割できているかを可視化するためのスクリプトです。

実行例：

```bash
python process_data/test_part_indices \
  --h5-path data/registrations_f.hdf5 \
  --sid 50020 \
  --seq jiggle_on_toes \
  --frame-id 0
```

主に以下を確認します。

- 各身体部位が正しく割り当てられているか
- 異なる身体部位の index が重複していないか
- 未ラベルの点が存在するか
- 可視化結果が妥当か

---

### 3. 部位分割用データセットの生成

`make_dfaust_part_dataset_public` は、ローカル環境で点群部位分割用サンプルを生成するためのスクリプトです。

実行例：

```bash
python process_data/make_dfaust_part_dataset_public \
  --male-h5 data/registrations_m.hdf5 \
  --female-h5 data/registrations_f.hdf5 \
  --save-root data/DFAUST_PART_SEG_ALL
```

生成されたデータセットは以下に保存されます。

```text
data/DFAUST_PART_SEG_ALL/
```

生成済みの完全なデータセットは GitHub にはアップロードしません。

---

## 対象モデル

生成されたラベルは、以下のような点群セグメンテーションモデルの学習・評価に利用できます。

- PointNet
- PointNet++
- DGCNN

本プロジェクトでは、モデル学習前のデータセット作成とラベル検証に重点を置いています。

---

## 評価予定

データセット作成後、以下の指標を用いてセグメンテーション性能を評価する予定です。

- Overall Accuracy
- Mean IoU
- Per-part IoU
- 欠損点群条件下での頑健性

---

## ライセンスに関する注意

元の DFAUST/FAUST データセットは、それぞれのライセンス条件に従います。

本リポジトリでは、元データセットおよび生成済みの完全なデータセットを再配布していません。  
公開しているのは、スクリプト、説明資料、可視化例のみです。
