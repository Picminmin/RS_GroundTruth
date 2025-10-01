
import os
from pathlib import Path
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import h5py
from pprint import pprint

__version__ = "0.0.1"
class RemoteSensingDataset:
    """リモートセンシングデータセットのロードと次元圧縮

    RemoteSensingDatasetをインスタンス化することで、読み込めるデータセットのキーワードを
    コンソールに表示する。
    ds = RemoteSensingDataset()
    X, y = ds.load(dataset_keyword)
    でデータセットの読み込みが完了する。
    X: (H, W, Bands)
    y: (H, W) となっている。

    """
    def __init__(self, base_dir=None, remove_bad_bands = True):
        """
        Args:
            base_dir (str): リモートセンシングデータのベースディレクトリ
                - Noneの場合は ~/.cache/RS_GroundTruth を利用
                - 明示的に与えた場合はそれを優先
            remove_bad_bands (bool): water absorption bandsを削除するオプション。
        """

        if base_dir is None:
            cache_dir = Path.home() / ".cache" / "RS_GroundTruth"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.base_dir = str(cache_dir)
        else:
            self.base_dir = base_dir

        self.remove_bad_bands = remove_bad_bands
        self.datasets = {}        # 読み込んだRSデータの(X, y)を保持するdict型の変数
        self.background_label = 0 # 背景ラベルが0であるようなRSデータセットを用いる
        self.available_data_keyword = [
            "Indianpines", "Salinas", "SalinasA", "Pavia", "PaviaU"
        ]

        print("[INFO] 利用可能なデータセットのキーワード:")
        for name in self.available_data_keyword:
            print(f" - {name}")

    def load(self, dataset_keyword):
        """
        指定したデータセットのキーワードを読み込み、特徴量とターゲットを返す
        Args:
            dataset_keyword (str): 読み込むデータセットのキーワード

        Returns:
            X (ndarray): 特徴量 (H, W, Bands)
            y (ndarray): ターゲット (H, W)
        """
        key_map = {
            "Indianpines": ("indian_pines", "indian_pines_gt"),
            "Salinas": ("salinas", "salinas_gt"),
            "SalinasA": ("salinasA", "salinasA_gt"),
            "Pavia": ("pavia", "pavia_gt"),
            "PaviaU": ("paviaU", "paviaU_gt"),
        }
        if dataset_keyword == "Indianpines":
            feature_path = os.path.join(self.base_dir, "01_Indian Pines/Indian_pines.mat")
            label_path = os.path.join(self.base_dir, "01_Indian Pines/Indian_pines_gt.mat")
            feature_key, label_key = key_map[dataset_keyword]

        elif dataset_keyword == "Salinas":
            feature_path = os.path.join(self.base_dir, "02_Salinas/Salinas.mat")
            label_path = os.path.join(self.base_dir, "02_Salinas/Salinas_gt.mat")
            feature_key, label_key = key_map[dataset_keyword]

        elif dataset_keyword == "SalinasA":
            feature_path = os.path.join(self.base_dir, "02_Salinas/SalinasA.mat")
            label_path = os.path.join(self.base_dir, "02_Salinas/SalinasA_gt.mat")
            feature_key, label_key = key_map[dataset_keyword]

        elif dataset_keyword == "Pavia":
            feature_path = os.path.join(self.base_dir, "03_Pavia Centre and University/Pavia.mat")
            label_path = os.path.join(self.base_dir, "03_Pavia Centre and University/Pavia_gt.mat")
            feature_key, label_key = key_map[dataset_keyword]

        elif dataset_keyword == "PaviaU":
            feature_path = os.path.join(self.base_dir, "03_Pavia Centre and University/PaviaU.mat")
            label_path = os.path.join(self.base_dir, "03_Pavia Centre and University/PaviaU_gt.mat")
            feature_key, label_key = key_map[dataset_keyword]

        # --- ファイル読み込み ---
        X_dict = load_mat_file(feature_path)
        y_dict = load_mat_file(label_path)

        X = np.array(X_dict[feature_key])
        y = np.array(y_dict[label_key])
        # --- bad bands を削除する場合 ---
        if dataset_keyword in ["Indianpines", "Salinas"] and self.remove_bad_bands:
            X_clean, _ = self.remove_noisy_and_absorption_bands(X=X, dataset_keyword=dataset_keyword)
            X = X_clean

        # datasetsにはloadした直近の1データセットのみ保持されるようにする。
        self.datasets.clear() # datasetsに保存する前にまずはクリアしておく。
        self.datasets[dataset_keyword] = {"X": X, "y": y}

        print(f"{dataset_keyword}を読み込みました。")
        return X, y

    def category_num(self, dataset_keyword):
        # 辞書オブジェクトself.datasetsにdataset_keywordキーがあるかを判定
        if dataset_keyword not in self.datasets:
            raise RuntimeError(f"{dataset_keyword} はロードされていません。")
        y = self.datasets[dataset_keyword]["y"]
        return len(np.unique(y[y != self.background_label])) # 背景=0 を除外

    # ================ 次元圧縮メソッド ================

    def apply_pca(self, X, n_components=30):
        """PCAによる次元圧縮
        ・ scikit-learn User Guide:
        https://scikit-learn.org/stable/modules/decomposition.html#pca
        """
        H, W, B = X.shape
        X_flat = X.reshape(-1, B)
        X_pca = PCA(n_components=n_components).fit_transform(X_flat)
        return X_pca.reshape(H, W, -1)

    def apply_ica(self, X, n_components=30):
        """ICAによる次元圧縮
        ・ scikit-learn User Guide:
        https://scikit-learn.org/stable/modules/decomposition.html#ica
        """
        H, W, B = X.shape
        X_flat = X.reshape
        X_ica = FastICA(n_components=n_components, random_state=0).fit_transform(X_flat)
        return X_ica.reshape(H, W, -1)

    def apply_lda(self, X, y, n_components=10):
        """LDAによる次元圧縮 (教師ラベル必要)
        ・ scikit-learn User Guide:
        https://scikit-learn.org/stable/modules/lda_qda.html
        """
        H, W, B = X.shape
        X_flat = X.reshape(-1, B)
        y_flat = y.flatten()
        mask = y_flat > 0 # 背景除去
        lda = LDA(n_components=n_components)
        X_lda = lda.fit_transform(X_flat[mask], y_flat[mask])
        # 背景はゼロ埋めに戻す
        X_out = np.zeros((X_flat.shape[0], n_components))
        X_out[mask] = X_lda
        return X_out.reshape(H, W, -1)

    def apply_tsne(self, X, n_components=2, sample_size = 5000):
        """t-SNEによる次元圧縮 (可視化用)
        ・ scikit-learn User Guide:
        https://scikit-learn.org/stable/modules/manifold.html#t-sne
        ・ 原著論文(Maaten & Hinton, 2008):
        ・ https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

        """
        H, W, B = X.shape
        X_flat = X.reshape(-1, B)
        # サンプル数が多いと遅いのでsubsample
        idx = np.random.choice(X_flat.shape[0], min(sample_size, X_flat.shape[0]), replace = False)
        X_sub = X_flat[idx]
        X_tsne = TSNE(n_components = n_components, rnadom_state = 0).fit_transform(X_sub)
        return X_tsne, idx

    def remove_noisy_and_absorption_bands(self, X, dataset_keyword):
        """
        Indian pines / Salinas系 のデータの水吸収帯域 + 端ノイズバンドを削除し、
        200バンドに統一する。

        Args:
            X (ndarray): 入力データ (H, W, Bands)
            dataset_keyword (str): "Indianpines" / "Salinas"

        Returns:
            X_clean (ndarray): 前処理後のデータ(H, W, 200)
            removed_bands (list): 削除したバンドのインデックス(0始まり)
        """
        # 削除するバンド (0始まりで指定)
        # AVIRIS (224 bands) 基準
        indianpines_bad_bands = list(range(0,4))        # 1-4
        indianpines_bad_bands += list(range(103,108))   # 104-108
        indianpines_bad_bands += list(range(149,163))   # 150-163
        indianpines_bad_bands += [219]                  # 220
        # AVIRIS (224 bands) 基準
        salinas_bad_bands = list(range(0,4))        # 1-4
        salinas_bad_bands += list(range(107,112))   # 108-112
        salinas_bad_bands += list(range(153,167))   # 154-167
        salinas_bad_bands += [223]                  # 224

        # マスクを作成
        mask = np.ones(X.shape[2], dtype = bool)
        if dataset_keyword == "Indianpines":
            # Indianpinesはすでに1~4を削除済み → バンド数220
            # つまり残っているのは"水吸収帯 20本"
            bad_bands = [i for i in indianpines_bad_bands
                         if i not in list(range(0,4))]
        elif dataset_keyword == "Salinas":
            bad_bands = salinas_bad_bands

        mask[bad_bands] = False
        X_clean = X[:, :, mask]
        print(f"[INFO] {dataset_keyword}: {X.shape[2]} → {X_clean.shape[2]} bands (removed {len(bad_bands)})")

        return X_clean, bad_bands

def load_mat_file(file_path):
    try:
        mat_data = scipy.io.loadmat(file_path)
        return {k: v for k, v in mat_data.items() if not k.startswith("__")}
    except Exception as e:
        print(f"[WARN] scipy.io.loadmat 失敗 ({e}), h5py に切り替えます")
        try:
            data = {}
            with h5py.File(file_path, "r") as f:
                for k in f.keys():
                    arr = np.array(f[k])
                    if arr.ndim > 1:
                        arr = arr.transpose()
                    data[k] = arr
            return data
        except Exception as e2:
            print(f"[ERROR] h5py でも失敗: {e2}")
            return {}


if __name__ == '__main__':

    # インスタンス化
    ds = RemoteSensingDataset()
    print(ds.available_data_keyword)

    dataset_keyword = "Indianpines"
    X, y = ds.load(dataset_keyword)
    print(f"[INFO] {dataset_keyword} category_num: {ds.category_num(dataset_keyword)}")
    # print(f"X type{type(X)}, y type:{type(y)}")
    print(X.shape)
    print(y.shape)
    print(f"[INFO] class_list: {list(np.unique(y))}")

    X_pca = ds.apply_pca(X=X, n_components=20)
    X_lda = ds.apply_lda(X=X, y=y, n_components=15)
    print(X_pca.shape)
    print(X_lda.shape)

    dataset_keyword = "Salinas"
    X, y = ds.load(dataset_keyword)
    print(X.shape)
    print(y.shape)
    print(f"[INFO] class_list: {list(np.unique(y))}")
    print(f"[INFO] {dataset_keyword} category_num: {ds.category_num(dataset_keyword)}")

    dataset_keyword = "SalinasA"
    X, y = ds.load(dataset_keyword)
    print(X.shape)
    print(y.shape)
    print(f"[INFO] class_list: {list(np.unique(y))}")
    print(f"[INFO] {dataset_keyword} category_num: {ds.category_num(dataset_keyword)}")

    dataset_keyword = "Pavia"
    X, y = ds.load(dataset_keyword)
    print(X.shape)
    print(y.shape)
    print(f"[INFO] class_list: {list(np.unique(y))}")
    print(f"[INFO] {dataset_keyword} category_num: {ds.category_num(dataset_keyword)}")

    dataset_keyword = "PaviaU"
    X, y = ds.load(dataset_keyword)
    print(X.shape)
    print(y.shape)
    print(f"[INFO] class_list: {list(np.unique(y))}")
    print(f"[INFO] {dataset_keyword} category_num: {ds.category_num(dataset_keyword)}")
