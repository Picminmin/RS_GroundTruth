
# RS_GroundTruth
利用可能なRSデータセットの読み込みと基本的な前処理(PCA, ICA, LDA, t-SNE)を提供する。
RSデータセットは['Hyperspectral Remote Sensing Scenes'][1]からダウンロードできます。

## 概要(Overview)

## インストール方法(Installation)
```bash
pip install git+https://github.com/Picminmin/RS_GroundTruth
```
インストールしたパッケージをGitHubリポジトリの最新版に置き換えたい場合には、以下のコマンドを実行してください。
```
pip install -U git+https://github.com/Picminmin/RS_GroundTruth
```

## 使い方(Usage / Examples)

```python
# RemoteSensingDatasetクラスのインポート
from RS_GroundTruth.rs_dataset import RemoteSensingDataset
# インスタンス化
ds = RemoteSensingDataset()
```
Console
```
[INFO] 利用可能なデータセットのキーワード:
 - Indianpines
 - Salinas
 - SalinasA
 - Pavia
 - PaviaU
```
```python
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


#### 注意
NAはNot Availableの略語で利用不可という意味である。
理由は安全にダウンロードできなかったためである。

#### anomaly detection(アノマリ検知、異常検知)
データセットの中から正常だとみなせないサンプルを特定する技術をanomaly detectionという。
'Xudong Kang's Homepage'(URL: 'https://xudongkang.weebly.com/data-sets.html') にデータセットが公開されているが、anomaly detectionを実現するためのデータセットの利用方法は掲載されていない。
