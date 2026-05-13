#!/bin/bash
# 下载 BGE 模型文件
# 运行一次即可, 之后 python 即可正常启动

MODEL_DIR="$(dirname "$0")/../models/bge-small-zh-v1.5"
mkdir -p "$MODEL_DIR"

echo "Downloading BAAI/bge-small-zh-v1.5 ..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='$MODEL_DIR')
print('Done.')
"
