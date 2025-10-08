from setuptools import setup, find_packages

setup(
    name="RS_GroundTruth",                # パッケージ名（pip install名にもなる）
    version="0.1.0",                      # バージョン
    description="Hyperspectral Remote Sensing Dataset Loader for Land-Cover Classification",
    author="Sohta Serikawa",
    license="MIT",
    packages=find_packages(include=["*"]),  # rs_dataset.py を含める
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "h5py",
    ],
    python_requires=">=3.8",
)
