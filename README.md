# ikuta-ml

## Prerequisites

* python = "^3.8"
* poetry

## Setup

```sh
poetry config virtualenvs.in-project true
poetry config --list

poetry install
```

## Job

* get_dataset_job.py
  * データ収集。csv形式で結果を保存
* preprocess_job.py
  * csvの前処理。pickleで結果を保存
* train_job.py
  * モデルの学習。SavedModel形式で推論用のモデルを保存

## Tips

* UbuntuでのMemoryErrorの回避

```sh
echo 1 | sudo tee /proc/sys/vm/overcommit_memory
```

* cudaのバージョン変更

```sh
ls -l /usr/local/ | grep cuda
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.0 /usr/local/cuda
ls -l /usr/local/ | grep cuda
```
