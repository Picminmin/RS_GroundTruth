
import os
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
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
    def __init__(self, base_dir="."):
        """
        Args:
            base_dir (str): リモートセンシングデータが格納されているディレクトリ. デフォルトは"."でカレントにする。
        """
        self.base_dir = base_dir
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
            feature_path = "./01_Indian Pines/Indian_pines.mat"
            label_path = "./01_Indian Pines/Indian_pines_gt.mat"
            feature_key, label_key = "indian_pines", "indian_pines_gt"

        elif dataset_keyword == "Salinas":
            feature_path = "./02_Salinas/Salinas.mat"
            label_path = "./02_Salinas/Salinas_gt.mat"
            feature_key, label_key = "salinas", "salinas_gt"

        elif dataset_keyword == "SalinasA":
            feature_path = "./02_Salinas/SalinasA.mat"
            label_path = "./02_Salinas/SalinasA_gt.mat"
            feature_key, label_key = "salinasA", "salinasA_gt"

        elif dataset_keyword == "Pavia":
            feature_path = "./03_Pavia Centre and University/Pavia.mat"
            label_path = "./03_Pavia Centre and University/Pavia_gt.mat"
            feature_key, label_key = "pavia", "pavia_gt"

        elif dataset_keyword == "PaviaU":
            feature_path = "./03_Pavia Centre and University/PaviaU.mat"
            label_path = "./03_Pavia Centre and University/PaviaU_gt.mat"
            feature_key, label_key = "paviaU", "paviaU_gt"

        X, y = load_mat_file(feature_path), load_mat_file(label_path)
        X, y = X[feature_key], y[label_key]
        print(f"{dataset_keyword}を読み込みました。")
        return X, y

    # ================ 次元圧縮メソッド ================

    def apply_pca(self, X, n_components=30):
        """PCAによる次元圧縮
        ・ scikit-learn User Guide:
        https://scikit-learn.org/stable/modules/decomposition.html#pca
        ・ 解説記事(統計的背景も含む日本語解説)
        https://qiita.com/nyk510/items/836293c3cbebcdee5fc2
        """
        H, W, B = X.shape
        X_flat = X.reshape(-1, B)
        X_pca = PCA(n_components=n_components).fit_transform(X_flat)
        return X_pca.reshape(H, W, -1)

    def apply_ica(self, X, n_components=30):
        """ICAによる次元圧縮
        ・ scikit-learn User Guide:
        https://scikit-learn.org/stable/modules/decomposition.html#ica
        ・ 日本語での応用解説(信号分離の例):
        https://data.gunosy.io/entry/ica
        """
        H, W, B = X.shape
        X_flat = X.reshape
        X_ica = FastICA(n_components=n_components, random_state=0).fit_transform(X_flat)
        return X_ica.reshape(H, W, -1)

    def apply_lda(self, X, y, n_components=10):
        """LDAによる次元圧縮 (教師ラベル必要)
        ・ scikit-learn User Guide:
        https://scikit-learn.org/stable/modules/lda_qda.html
        ・ 日本語解説(数学的導出含む):
        https://analytics-note.xyz/machine-learning/lda-overview/
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
        ・ 日本語の図解解説:
        https://data-analytics.fun/2021/04/05/tsne-intro/

        """
        H, W, B = X.shape
        X_flat = X.reshape(-1, B)
        # サンプル数が多いと遅いのでsubsample
        idx = np.random.choice(X_flat.shape[0], min(sample_size, X_flat.shape[0]), replace = False)
        X_sub = X_flat[idx]
        X_tsne = TSNE(n_components = n_components, rnadom_state = 0).fit_transform(X_sub)
        return X_tsne, idx

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
