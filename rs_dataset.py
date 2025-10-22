
import os
from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
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
        X,
        n_components=None,
        variance_threshold: float = 0.98,
        mean_centering=True,
        whitening_option = "pca_whitening",
        eps = 1E-6,
        dataset_keyword=None,
        save_dir = "img"
    ):
        """PCAによる次元圧縮(寄与率プロット付き・動的主成分数決定)

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
            X (ndarray): 入力データ(H, W, B)
            n_components (int or None): 固定主成分数。Noneなら累積寄与率から自動決定。
            variance_threshold (float): 累積寄与率がこの値以上になるような最小の主成分数を選ぶ。
            mean_centering (bool): 平均中心化をするかどうか。
            whitening_option (str): 相関を消して、各軸の分散を1(無相関かつ等分散)に
            そろえる線形変換を行う白色化。
            ["pca_whitening", "zca_whitening"]のいずれかが選べる。
            デフォルトは、pca_whitening。pca_whiteningがPCAで得られた主成分軸に対してデータを
            プロットするのに対し、zca_whiteningはもとのは座標系上にデータをプロットする。
            PCA, ZCAそれぞれでのRS画像による土地被覆分類手法に与えるPCA, ZCAの効果の違いが
            報告された比較研究は現時点では見当たらない。また、土地被覆分類手法の文脈では
            PCAやMNF(Minimum Noise Fraction)が前処理として頻出するため、デフォルトの
            pca_whiteningを使うことにする。
            eps (float): 数値安定化項。固有ベクトルを要素に持つ対角行列が発散しないようにするため
            の小さい値。
            dataset_keyword (str): データセット名 (例: "Indianpines")。出力ファイル名に使用。
            save_dir (str): 寄与率プロットの保存先ディレクトリ。

        Returns:
            ndarray: 変換後データ(H, W, n_components_used)
        """

        # --- 1. 前処理 ---
        H, W, B = X.shape
        X_flat = X.reshape(-1, B) # (n_samples, n_features)=(N, D)
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
        idx = np.argsort(eigvals)[::-1] # 固有値を大きい順に並べ替え
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # --- 4. 寄与率と累積寄与率 ---
        explained_var_ratio = eigvals / np.sum(eigvals)
        cumulative_ratio = np.cumsum(explained_var_ratio)

        # --- 5. 主成分数を自動決定 ---
        if n_components is None:
            n_components = np.searchsorted(cumulative_ratio, variance_threshold) + 1

        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

        # --- 6. whitening ---
        if whitening_option == "pca_whitening":
            P_pca = np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
            U_pca = Xc @ P_pca.T
            U_pca = U_pca.reshape((H, W, n_components))
        elif whitening_option == "zca_whitening":
            P_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
            U_pca = Xc @ P_zca.T
            U_pca = U_pca.reshape((H, W, n_components))
        else:
            raise ValueError("whitening_option must be 'pca_whitening' or 'zca_whitening'")

        # --- 7. 属性に保存 ---
        self.n_components_used = n_components
        self.explained_variance_ratio_ = explained_var_ratio
        self.cumulative_variance_ratio_ = cumulative_ratio
        self.variance_threshold = variance_threshold

        print(f"[INFO] PCA: 寄与率閾値 {variance_threshold:.2f} に対して"
              f"{n_components} 成分を採用 (累積寄与率={cumulative_ratio[n_components-1]:.4f})")

        # --- 8. プロット ---
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"pca_variance_{dataset_keyword or 'RSdataset'}.png")

        plt.figure(figsize=(6,4))
        plt.plot(np.arange(1,len(explained_var_ratio) + 1), explained_var_ratio,
                 marker="o",label="Explained variance ratio")
        plt.plot(np.arange(1,len(cumulative_ratio) + 1), cumulative_ratio,
                 marker="s",label="Cumulative variance ratio")
        plt.axhline(variance_threshold, color="r", linestyle="--",
                    label=f"Threshold = {variance_threshold:.2f}" )
        plt.axhline(n_components, color="g", linestyle="--",
            label=f"n_components = {n_components:.2f}" )
        plt.xlabel("Principal Component Index")
        plt.ylabel("Variance Ratio")
        plt.title(f"PCA Variance Analysis ({dataset_keyword})")
        # 表示する値の範囲の指定
        plt.xlim(0,n_components + 10) # 横軸の範囲の指定(0からn_componentsに適当な定数を加えた数まで)
        plt.ylim(0,1) # 縦軸の範囲の指定(0から1まで)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(save_path, dpi = 300)
        plt.close()
        print(f"[INFO] 寄与率プロットを保存しました → {save_path}")
        return U_pca

    def apply_pca_diagnostic(
        self,
        X,
        n_components=None,
        variance_threshold: float = 0.995,
        mean_centering=True,
        whitening_option="pca_whitening",
        eps=1e-6,
        dataset_keyword=None,
        save_dir="img"
    ):
        """
        PCAによる次元圧縮 + 各主成分のスペクトル的意味の可視化

        ・寄与率プロット
        ・主成分ごとのバンド寄与 (loading plot)
        ・元バンドとの相関ヒートマップ

        Args:
            X (ndarray): 入力データ (H, W, B)
            n_components (_type_, optional): 固定主成分数。Noneなら累積寄与率で自動決定。
            variance_threshold (float, optional): 累積寄与率がこの値以上になるように主成分数を選択する
            mean_centering (bool): 平均中心化するか
            whitening_option (str): "pca_whitening" or "zca_whitening"
            eps (float, optional): 数値安定化項
            dataset_keyword (str, optional): データセット名
            save_dir (str, optional): 保存ディレクトリ
        Returns:
            ndarray: 変換後データ (H, W, n_components_used)
        """

        os.makedirs(save_dir, exist_ok=True)
        H, W, B = X.shape
        X_flat = X.reshape(-1, B)

        # --- 1. 平均中心化 ---
        if mean_centering:
            mean = np.mean(X_flat, axis=0)
            Xc = X_flat - mean
        else:
            Xc = X_flat

        # --- 2. 共分散行列 & 固有値分解 ---
        N = X_flat.shape[0]
        cov = (1.0 / N) * Xc.T @ Xc
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # --- 3. 寄与率 ---
        explained_var_ratio = eigvals / np.sum(eigvals)
        cumulative_ratio = np.cumsum(explained_var_ratio)

        # --- 4. 主成分数を自動決定 ---
        if n_components is None:
            n_components = np.searchsorted(cumulative_ratio, variance_threshold) + 1
        eigvals, eigvecs = eigvals[:n_components], eigvecs[:, :n_components]

        # --- 5. whitening (既存処理) ---
        if whitening_option == "pca_whitening":
            P_pca = np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
            U_pca = Xc @ P_pca.T
            U_pca = U_pca.reshape((H, W, n_components))
        elif whitening_option == "zca_whitening":
            P_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
            U_pca = Xc @ P_zca.T
            U_pca = U_pca.reshape((H, W, n_components))
        else:
            raise ValueError("whitening_option must be 'pca_whitening' or 'zca_whitening'")

        # --- 6. 基本情報出力 ---
        print(f"[INFO] PCA: 累積寄与率 {cumulative_ratio[n_components-1]:.4f} で {n_components} 成分を採用")
        print(f"[INFO] 第一主成分 ~ 第{n_components}主成分の寄与率: {explained_var_ratio[:n_components]}")

        # --- 7. 寄与率プロット ---
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(explained_var_ratio) + 1), explained_var_ratio, "o-", label="Explained variance")
        plt.plot(np.arange(1, len(cumulative_ratio) + 1), cumulative_ratio, "s-", label="Cumulative variance")
        plt.axhline(variance_threshold, color="r", linestyle="--", label=f"Threshold={variance_threshold:.2f}")
        plt.xlabel("Principal Component Index")
        plt.ylabel("Variance Ratio")
        plt.title(f"PCA Variance ({dataset_keyword})")
        plt.legend()
        plt.tight_layout()
        save_path_var = os.path.join(save_dir, f"pca_variance_{dataset_keyword or 'RS'}.png")
        plt.savefig(save_path_var, dpi = 300)
        plt.close()
        print(f"[INFO] 寄与率プロットを保存しました → {save_path_var}")

        # --- 8. 主成分のバンド寄与 (Loading Plot) ---
        plt.figure(figsize=(7, 4))
        for i in range(n_components):
            plt.plot(np.arange(B), eigvecs[:, i], label=f"PC{i+1} ({explained_var_ratio[i]*100:.1f}%)")
        plt.xlabel("Band index (wavelength order)")
        plt.ylabel("Loading weight")
        plt.title(f"PCA Band Loadings ({dataset_keyword})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path_load = os.path.join(save_dir, f"pca_loadings_{dataset_keyword or 'RS'}.png")
        plt.savefig(save_path_load, dpi = 300)
        plt.close()
        print(f"[INFO] 主成分ベクトルプロットを保存しました → {save_path_load}")

        # --- 9. 元バンドとの相関ヒートマップ ---
        X_flat_norm = (X_flat - X_flat.mean(axis=0)) / (X_flat.std(axis=0) + eps)
        U_flat_norm = (U_pca.reshape(-1, n_components) - U_pca.reshape(-1, n_components).mean(axis=0)) \
                       / (U_pca.reshape(-1, n_components).std(axis=0) + eps)
        corr_matrix = np.corrcoef(X_flat_norm.T, U_flat_norm.T)[0:B, B:]

        plt.figure(figsize=(6, 5))
        plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation coefficient")
        plt.xlabel("Principal Component")
        plt.ylabel("Band index")
        plt.title(f"Correlation: Bands vs PCA Components ({dataset_keyword})")
        plt.tight_layout()
        save_path_corr = os.path.join(save_dir, f"pca_corr_{dataset_keyword or 'RS'}.png")
        plt.savefig(save_path_corr, dpi = 300)
        plt.close()
        print(f"[INFO] バンド・主成分相関ヒートマップを保存しました → {save_path_corr}")

        # --- 10. 結果を属性に保存 ---
        self.pca_eigvecs_ = eigvecs
        self.pca_eigvals_ = eigvals
        self.pca_explained_var_ratio_ = explained_var_ratio
        self.pca_corr_matrix_ = corr_matrix
        self.n_components_used = n_components

        return U_pca



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

    # dataset_keyword = "Salinas"
    # X, y = ds.load(dataset_keyword)
    # print(X.shape)
    # print(y.shape)
    # print(f"[INFO] class_list: {list(np.unique(y))}")
    # print(f"[INFO] {dataset_keyword} category_num: {ds.category_num(dataset_keyword)}")
#
    # dataset_keyword = "SalinasA"
    # X, y = ds.load(dataset_keyword)
    # print(X.shape)
    # print(y.shape)
    # print(f"[INFO] class_list: {list(np.unique(y))}")
    # print(f"[INFO] {dataset_keyword} category_num: {ds.category_num(dataset_keyword)}")
#
    # dataset_keyword = "Pavia"
    # X, y = ds.load(dataset_keyword)
    # print(X.shape)
    # print(y.shape)
    # print(f"[INFO] class_list: {list(np.unique(y))}")
    # print(f"[INFO] {dataset_keyword} category_num: {ds.category_num(dataset_keyword)}")
#
    # dataset_keyword = "PaviaU"
    # X, y = ds.load(dataset_keyword)
    # print(X.shape)
    # print(y.shape)
    # print(f"[INFO] class_list: {list(np.unique(y))}")
    # print(f"[INFO] {dataset_keyword} category_num: {ds.category_num(dataset_keyword)}")
