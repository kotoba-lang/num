#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == "Darwin" ]] || { echo "native Metal verification requires macOS" >&2; exit 2; }
cmake -S native/metal -B target/metal -DCMAKE_BUILD_TYPE=Release
cmake --build target/metal --parallel
clojure -J-Djna.library.path="$PWD/target/metal" -M:metal-native-verify
clojure -M:metal-verify
