
import os
import subprocess
import requests
from pathlib import Path
import time
from shutil import which

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

def has_command(cmd: str) -> bool:
    """コマンドが存在するか確認"""
    return which(cmd) is not None



def robust_download(url, filename, retries=3, delay=5):
    """堅牢なダウンロード処理 (aria2c > wget > Python requests) """

    # aria2c があれば優先
    if has_command("aria2c"):
        print("[INFO] Using aria2c for download")
        cmd = ["aria2c", "-x", "8", "-s", "8", "-c", "-o", filename, url]
        subprocess.run(cmd, check=True)
        return

    # wget があれば次に使う
    if has_command("wget"):
        print("[INFO] Using wget for download")
        cmd = ["wget", "-c", "-O", filename, url]
        subprocess.run(cmd, chech=True)
        return

    # fallback: Python requests
    print("[INFO] Using Python requests fallback (less robust)")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) # 総サイズ (bytes)
                downloaded = 0
                start_time = time.time()
                next_report = 10 # 次に経過時間を表示する閾値 (秒)

                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536): # 64KB
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # 経過時間をチェック
                            elapsed = int(time.time() - start_time)
                            if elapsed >= next_report:
                                if total > 0:
                                    percent = downloaded / total * 100
                                    print(f"[INFO] Downloading... elapsed {elapsed} sec ({percent:.2f}%)")
                                else:
                                    print(f"[INFO] Downloading... elapsed {elapsed} seconds")
                                next_report += 10
                # ✅ 完了時に 100% を明示
                elapsed = int(time.time() - start_time)
                if total > 0:
                    print(f"[INFO] Download complete! elapsed {elapsed} sec (100.00%)")
            return # 成功したら終了
        except Exception as e:
            print(f"[WARN] Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"[INFO] Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                # リトライ回数を越えたら例外
                raise RuntimeError(f"Failed to download {url} after {retries} attempts")


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

        # 不完全なファイルが残っていたら削除
        if os.path.exists(file_path):
            try:
                expected_size = int(requests.head(url, timeout=10).headers.get("Content-Length", 0))
                local_size = os.path.getsize(file_path)
                if expected_size > 0 and local_size < expected_size:
                    print(f"[WARN] Incomplete file detected. Deleting {file_path}")
                    os.remove(file_path)
            except Exception:
                pass # Content-Length が取れない場合は無視

        if not os.path.exists(file_path):
            print(f"[INFO] Downloading {filename} ...")
            robust_download(url, file_path)
        else:
            print(f"[INFO] {filename} already exists. Skipping download. ")

        paths[key] = file_path

    return paths

if __name__ == "__main__":
    dataset_keyword = "Pavia"
    fetch_dataset(dataset_keyword)
