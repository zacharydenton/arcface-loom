#!/usr/bin/env bash
# Build the command-line tools and the resident C ABI used by arcface_loom.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

hipcc="${HIPCC:-/opt/rocm/bin/hipcc}"
mkdir -p build

"$hipcc" -O2 -Wall -Werror -fPIC -shared -DARCFACE_LIBRARY -o build/libarcface.so host/arcface.cpp
"$hipcc" -O2 -Wall -Werror -o host/arcface host/arcface.cpp
"$hipcc" -O2 -Wall -Werror -o host/loomrun host/loomrun.cpp

printf 'built %s\n' build/libarcface.so host/arcface host/loomrun
