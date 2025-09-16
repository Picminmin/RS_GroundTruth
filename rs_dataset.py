
import os
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

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
                - Noneの場合は自動判定
                - 明示的に与えた場合はそれを優先
            remove_bad_bands (bool): water absorption bandsを削除するオプション。
                                     デフォルトはTrueとする。
        """

        if base_dir is None:
            cwd = os.getcwd()
            if "RS_GroundTruth" in cwd:
                # RS_GroundTruthの中から呼ばれたとき
                self.base_dir = "." # カレントに指定
            else:
                # main/ など外側から呼ばれたとき
                self.base_dir = "./RS_GroundTruth"
        else:
            self.base_dir = base_dir

        self.remove_bad_bands = remove_bad_bands
        self.available_data_keyword = ["Indianpines", "Salinas", "SalinasA", "Pavia", "PaviaU"]

        # インスタンス化時に利用可能なデータセットのキーワードを表示
        if self.available_data_keyword:
            print("[INFO] 利用可能なデータセットのキーワード:")
            for name in self.available_data_keyword:
                print(f" - {name}")
        else:
            print("[WARN] 利用可能なデータセットはありません")

    def load(self, dataset_keyword):
        """
        指定したデータセットのキーワードを読み込み、特徴量とターゲットを返す
        Args:
            dataset_keyword (str): 読み込むデータセットのキーワード

        Returns:
            X (ndarray): 特徴量 (H, W, Bands)
            y (ndarray): ターゲット (H, W)
        """
        if dataset_keyword == "Indianpines":
            feature_path = os.path.join(self.base_dir, "01_Indian Pines/Indian_pines.mat")
            label_path = os.path.join(self.base_dir, "01_Indian Pines/Indian_pines_gt.mat")
            feature_key, label_key = "indian_pines", "indian_pines_gt"

        elif dataset_keyword == "Salinas":
            feature_path = os.path.join(self.base_dir, "02_Salinas/Salinas.mat")
            label_path = os.path.join(self.base_dir, "02_Salinas/Salinas_gt.mat")
            feature_key, label_key = "salinas", "salinas_gt"

        elif dataset_keyword == "SalinasA":
            feature_path = os.path.join(self.base_dir, "02_Salinas/SalinasA.mat")
            label_path = os.path.join(self.base_dir, "02_Salinas/SalinasA_gt.mat")
            feature_key, label_key = "salinasA", "salinasA_gt"

        elif dataset_keyword == "Pavia":
            feature_path = os.path.join(self.base_dir, "03_Pavia Centre and University/Pavia.mat")
            label_path = os.path.join(self.base_dir, "03_Pavia Centre and University/Pavia_gt.mat")
            feature_key, label_key = "pavia", "pavia_gt"

        elif dataset_keyword == "PaviaU":
            feature_path = os.path.join(self.base_dir, "03_Pavia Centre and University/PaviaU.mat")
            label_path = os.path.join(self.base_dir, "03_Pavia Centre and University/PaviaU_gt.mat")
            feature_key, label_key = "paviaU", "paviaU_gt"

        # --- ファイル読み込み ---
        X_dict, y_dict = load_mat_file(feature_path), load_mat_file(label_path)
        X, y = np.array(X_dict[feature_key]), np.array(y_dict[label_key])

        # --- bad bands を削除する場合 ---
        if dataset_keyword in ["Indianpines", "Salinas"] and self.remove_bad_bands:
                X_clean, _ = self.remove_noisy_and_absorption_bands(X=X, dataset_keyword=dataset_keyword)

        print(f"{dataset_keyword}を読み込みました。")
        return X, y

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
    """MATファイルの読み込み"""
    try:
        mat_data = scipy.io.loadmat(file_path)
        # 不要なメタデータを除去
        mat_data_cleaned = {key: value for key, value in mat_data.items()
                            if not key.startswith('__')}
        return mat_data_cleaned
    except FileNotFoundError:
        print(f'エラー: ファイルが見つかりません:{file_path}')
    except Exception as e:
        print(f'エラーが発生しました:{e}')

if __name__ == '__main__':

    # インスタンス化
    ds = RemoteSensingDataset(base_dir=".")
    X, y = ds.load("Indianpines")
    print(X.shape)
    print(y.shape)

    X, y = ds.load("Salinas")
    print(X.shape)
    print(y.shape)

    X, y = ds.load("SalinasA")
    print(X.shape)
    print(y.shape)

    X, y = ds.load("Pavia")
    print(X.shape)
    print(y.shape)

    X, y = ds.load("PaviaU")
    print(X.shape)
    print(y.shape)
