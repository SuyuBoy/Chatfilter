#!/bin/bash
# Cython 编译热路径模块为 .so
# IO 层不参与编译 (http_server.py, demo_terminal_server.py, sender.py)
set -e

cd "$(dirname "$0")/.."

MODULES=(
  src/engine/simhash_dedup.py
  src/engine/micro_cluster.py
  src/engine/cycle_compressor.py
  src/engine/preprocessor.py
)

echo "Cython 编译热路径模块..."
python -m Cython.Build.cythonize -3 -i "${MODULES[@]}"

echo ""
echo "✅ 编译完成"
find src/engine -name '*.so' -exec ls -lh {} \;
