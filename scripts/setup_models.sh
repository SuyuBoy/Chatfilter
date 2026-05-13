#!/bin/bash
# 下载 BGE 模型文件
# 运行一次即可, 之后 python 即可正常启动

ROOT="$(dirname "$0")/.."

download() {
  local name=$1 dir="$ROOT/models/$name"
  mkdir -p "$dir"
  echo "Downloading BAAI/$name ..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/$name', local_dir='$dir')
print('Done.')
"
}

# 默认下载 small (100MB, 快速验证)
download "bge-small-zh-v1.5"

# 可选下载 base (400MB, 生产推荐, 聚类+4分)
if [ "$1" = "--all" ] || [ "$1" = "--base" ]; then
  download "bge-base-zh-v1.5"
fi

echo ""
echo "All models ready. In config/settings.py:"
echo "  model_name='models/bge-small-zh-v1.5'   # 快速验证 (默认)"
echo "  model_name='models/bge-base-zh-v1.5'    # 生产推荐"
