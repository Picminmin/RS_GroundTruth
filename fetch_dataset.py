
import os
import urllib.request
from pathlib import Path

# 各データセットのURLとサブディレクトリ
DATASET_URLS = {
    "Indianpines": {
        "feature": "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat",
        "label": "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        "subdir": "01_Indian Pines",
    },
    "Salinas": {
        "feature": "http://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat",
        "label": "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        "subdir": "02_Salinas",
    },
    "SalinasA": {
        "feature": "http://www.ehu.eus/ccwintco/uploads/f/fa/SalinasA.mat",
        "label": "http://www.ehu.eus/ccwintco/uploads/f/fa/SalinasA_gt.mat",
        "subdir": "02_Salinas",
    },
    "Pavia": {
        "feature": "http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat",
        "label": "http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat",
        "subdir": "03_Pavia Centre and University",
    },
    "PaviaU": {
        "feature": "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "label": "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        "subdir": "03_Pavia Centre and University",
    },
}

def fetch_dataset(name: str, base_dir: str = None):
    """
    指定されたデータセットをEHUサイトからダウンロードして保存する。
    デフォルトでは ~/.cache/RS_GroundTruth/ に保存。
    """
    if name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {name}")

    # base_dir が未指定ならキャッシュディレクトリを使う
    if base_dir is None:
        cache_dir = Path.home() / ".cache" / "RS_GroundTruth"
        cache_dir.mkdir(parents=True, exist_ok=True)
        base_dir = str(cache_dir)

    info = DATASET_URLS[name]
    dataset_dir = os.path.join(base_dir, info["subdir"])
    os.makedirs(dataset_dir, exist_ok=True)

    urls = {"feature": info["feature"], "label": info["label"]}
    paths = {}

    for key, url in urls.items():
        filename = os.path.basename(url)
        file_path = os.path.join(dataset_dir, filename)

        if not os.path.exists(file_path):
            print(f"[INFO] Downloading {filename} ...")
            urllib.request.urlretrieve(url, file_path)
        else:
            print(f"[INFO] {filename} already exists. Skipping download. ")

        paths[key] = file_path

    return paths
