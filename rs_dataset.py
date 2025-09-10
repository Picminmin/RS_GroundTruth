
import numpy as np
import scipy.io

class RemoteSensingDataset:
    """
    RemoteSensingDatasetをインスタンス化することで、読み込めるデータセットのキーワードを
    コンソールに表示する。
    ds = RemoteSensingDataset()
    X, y = ds.load(dataset_keyword)
    でデータセットの読み込みが完了する。
    X: (H, W, Bands)
    y: (H, W) となっている。

    このクラスに、主成分分析も実装したい。
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
