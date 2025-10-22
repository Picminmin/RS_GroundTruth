
# RS_GroundTruth
利用可能なRSデータセットの読み込みと基本的な前処理(PCA, ICA, LDA, t-SNE)を提供する。

<!--
RSデータセットは['Hyperspectral Remote Sensing Scenes'][1]でダウンロードできるものを使用しています。
RSデータセットは、本パッケージ内にあるfetch_dataset()を実行することでダウンロードされ、利用できるようになります。
-->


## 概要(Overview)

## インストール方法(Installation)
```bash
pip install -e git+https://github.com/Picminmin/RS_GroundTruth
```
インストールしたパッケージをGitHubリポジトリの最新版に置き換えたい場合には、以下のコマンドを実行してください。
```bash
pip install -U git+https://github.com/Picminmin/RS_GroundTruth
```

## データ利用とクレジット

このパッケージでは以下のハイパースペクトルリモートセンシングデータセットを利用しています：

- Indian Pines  
- Salinas  
- SalinasA  
- Pavia Centre  
- Pavia University  

これらのデータは、**Universidad del País Vasco / Euskal Herriko Unibertsitatea (UPV/EHU), Grupo de Inteligencia Computacional (ccwintco group)** が公開している  
「['Hyperspectral Remote Sensing Scenes'][1]」サイトから取得したものです。  

ご利用の際は、上記サイトをクレジットとして明示するとともに、各データセットを最初に公開した研究・論文を適切に引用してください。例：  

- **Indian Pines**: AVIRIS sensor, NASA Jet Propulsion Laboratory.  
- **Salinas / SalinasA**: AVIRIS sensor over Salinas Valley, California.  
- **Pavia Centre / Pavia University**: ROSIS-03 sensor over Pavia, Italy.  

> 参考文献の一例:  
> Pavia University scene — “Pavia University scene was acquired by the ROSIS-03 sensor over Pavia, Italy”  
> (see: [Optical Engineering / Applied Optics](https://opg.optica.org/abstract.cfm?uri=ao-59-13-4151))

本パッケージは、データ配布元の EHU グループとは独立に開発されています。


# RS_GroundTruth データセットの配置方法

## ダウンロード元
各データセットは University of the Basque Country (EHU) の公式サイトから入手してください：  
👉 [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)

## ディレクトリ構成
ダウンロードした `.mat` ファイルは、以下のような構成に配置してください。

```text
RS_GroundTruth/
├── 01_Indian Pines/
│   ├── Indian_pines.mat
│   └── Indian_pines_gt.mat
│
├── 02_Salinas/
│   ├── Salinas.mat
│   ├── Salinas_gt.mat
│   ├── SalinasA.mat
│   └── SalinasA_gt.mat
│
├── 03_Pavia Centre and University/
│   ├── Pavia.mat
│   ├── Pavia_gt.mat
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
│
└── rs_dataset.py
```

## 使い方(Usage / Examples)
<!--
from RS_GroundTruth import fetch_dataset, RemoteSensingDataset
# データをダウンロード( ~/.cache/RS_GroundTruth に保存される)
# ここで、~はシェル上でのユーザーのホームディレクトリを指す記号である。
fetch_dataset("Indianpines")　# 他のdataset_keywordも同様にしてダウンロードしてください
-->


```python
# RemoteSensingDatasetクラスのインポート
from RS_GroundTruth import RemoteSensingDataset
# インスタンス化
ds = RemoteSensingDataset(base_dir="RS_GroundTruthのpath") # ← ローカルPCの中にあるRS_GroundTruthのpathを入力してください
print(ds.available_data_keyword) # ['Indianpines', 'Salinas', 'SalinasA', 'Pavia', 'PaviaU'] # ← dataset_keywordに入力できる値
```
Console
```console
[INFO] 利用可能なデータセットのキーワード:
 - Indianpines
 - Salinas
 - SalinasA
 - Pavia
 - PaviaU
```

```python
# Indianpinesの読み込み
X, y = ds.load("Indianpines")
print(X.shape) # (145, 145, 200)
print(y.shape) # (145, 145)

# pcaによる次元圧縮
X_pca = ds.apply_pca(X=X, n_components=20)
print(X_pca.shape) # (145, 145, 20)

# ldaによる次元圧縮
X_lda = ds.apply_lda(X=X, y=y, n_components=15)
print(X_lda.shape) # (145, 145, 15)
```



<!-- 参考文献 -->
[1]:https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes



<!--
### データセットのダウンロードについて

`fetch_dataset` 関数は大容量ファイルを安定してダウンロードするため、
環境に `aria2c` または `wget` があれば自動的に利用します。
インストールされていない場合は Python の requests にフォールバックしますが、
通信環境によっては失敗することがあります。

推奨:  
- Linux/macOS: `sudo apt install aria2` または `brew install aria2`  
- Windows: [aria2 release page](https://github.com/aria2/aria2/releases) からバイナリを入手
-->


