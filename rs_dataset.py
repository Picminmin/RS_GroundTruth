
import os
from pathlib import Path
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
# import h5py
from pprint import pprint
from pathlib import Path

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
            # 一つ上の階層のディレクトリパスを取得
            main_project_dir = Path(__file__).parent.parent.resolve()
            used_dir = main_project_dir / "RS_GroundTruth"
            self.base_dir = str(used_dir)
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

    def apply_pca(
        self,
        X, n_components=30,
        mean_centering=True,
        whitening_option = "pca_whitening",
        eps = 1E-6
        ):
        """PCAによる次元圧縮
        ・ scikit-learn User Guide:
        https://scikit-learn.org/stable/modules/decomposition.html#pca

        平均中心化(mean-centering)によるPCAを実装する。平均中心化をしない
        場合は主成分がデータの平均ベクトルに影響を受けるために、平均中心化
        をしなかった理由をきつく詰められると考えられる。
        PCAにおいて、平均中心化をしておくのはお作法という位置づけと思っても
        よいかもしれない。
        ・ "The Effect of Data Centering on PCA Models"
        URL: https://eigenvector.com/wp-content/uploads/2020/06/EffectofCenteringonPCA.pdf

        Args:
            mean_centering (bool): 平均中心化をするかどうか。デフォルトはTrue。
            whitening_option (str): 相関を消して、各軸の分散を1(無相関かつ等分散)に
            そろえる線形変換を行う白色化。
            ["pca_whitening", "zca_whitening"]のいずれかが選べる。
            デフォルトは、pca_whitening。pca_whiteningがPCAで得られた主成分軸に対してデータを
            プロットするのに対し、zca_whiteningはもとのは座標系上にデータをプロットする。
            PCA, ZCAそれぞれでのRS画像による土地被覆分類手法に与えるPCA, ZCAの効果の違いが
            報告された比較研究は現時点では見当たらない。また、土地被覆分類手法の文脈では
            PCAやMNF(Minimum Noise Fraction)が前処理として頻出するため、デフォルトの
            pca_whiteningを使うことにする。
            eps (float): 固有ベクトルを要素に持つ対角行列が発散しないようにするための
            小さい値。
        """
        H, W, B = X.shape
        X_flat = X.reshape(-1, B) # (n_samples, n_features)=(N, D)

        # --- 1. 平均中心化 ---
        if mean_centering:
            mean = np.mean(X_flat, axis=0)
            Xc = X_flat - mean
        else:
            Xc = X_flat

        # --- 2. 共分散行列 Φ_X = (1/N) X X^T ---
        N = X_flat.shape[0]
        cov = (1.0 / N) * Xc.T @ Xc # (D, D)

        # --- 3. 固有値分解 Φ_X = A Ω A^T ---
        eigvals, eigvecs = np.linalg.eigh(cov) # 対称行列なのでeighが安定
        # 固有値を大きい順に並べ替え
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # --- 上位 n_components だけ残す ---
        if n_components is not None and n_components < eigvals.shape[0]:
            eigvals = eigvals[:n_components]
            eigvecs = eigvecs[:, :n_components]

        # --- 4. Whitening行列 ---
        # --- P_PCA = Ω^{-1/2} A^T ---
        if whitening_option == "pca_whitening":
            P_pca = np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
            # U_pca: 平均中心化データ行列へのP_pcaによる線形変換
            U_pca = Xc @ P_pca.T # ← shape: (N, n_components)
            U_pca = U_pca.reshape((H, W, n_components))
            return U_pca
        elif whitening_option == "zca_whitening":
            P_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
            U_zca = Xc @ P_zca.T # ← shape: (N, n_components)
            U_zca = U_zca.reshape((H, W, n_components))
            return U_zca

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
        print(f"[WARN] scipy.io.loadmat 失敗 ({e})")
        # try:
            # data = {}
            # with h5py.File(file_path, "r") as f:
                # for k in f.keys():
                    # arr = np.array(f[k])
                    # if arr.ndim > 1:
                        # arr = arr.transpose()
                    # data[k] = arr
            # return data
        # except Exception as e2:
            # print(f"[ERROR] h5py でも失敗: {e2}")
            # return {}

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
    print(X_pca.shape)
    # X_lda = ds.apply_lda(X=X, y=y, n_components=15)
    # print(X_lda.shape)

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
