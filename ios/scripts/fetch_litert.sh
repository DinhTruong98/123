#!/bin/sh
# Download the official LiteRT (TensorFlow Lite) C library for iOS and repackage it as a static
# library XCFramework in Packages/LiteRTC/TensorFlowLiteC.xcframework (git-ignored).
#
# Same binary as the official LiteRT Swift package (github.com/google-ai-edge/LiteRT, tag v2.3.2,
# Package.swift: URL + SHA-256 below). Repackaged because Xcode embeds the official static
# *framework* as if it were dynamic and then rejects it (no Info.plist); a library XCFramework
# (.a + headers) is linked statically and never embedded.
set -eu

URL=https://github.com/google-ai-edge/LiteRT/releases/download/v2.3.0/TensorFlowLiteC.xcframework.zip
SHA256=f77274f9249af1e7dd1de88f57661944f80d177ff2c386b28aa3d466b7173f50

cd "$(dirname "$0")/../Packages/LiteRTC"
OUT=TensorFlowLiteC.xcframework
if [ -f "$OUT/.sha256" ] && [ "$(cat "$OUT/.sha256")" = "$SHA256" ]; then
    echo "$OUT is up to date"
    exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fL --retry 3 -o "$TMP/litert.zip" "$URL"
echo "$SHA256  $TMP/litert.zip" | shasum -a 256 -c -
unzip -q "$TMP/litert.zip" -d "$TMP/src"

ARGS=""
for SLICE in ios-arm64 ios-arm64-simulator; do
    FW="$TMP/src/TensorFlowLiteC.xcframework/$SLICE/TensorFlowLiteC.framework"
    D="$TMP/$SLICE"
    mkdir -p "$D/Headers/TensorFlowLiteC"
    cp "$FW/TensorFlowLiteC" "$D/libTensorFlowLiteC.a"
    cp "$FW"/Headers/*.h "$D/Headers/TensorFlowLiteC/"
    cat > "$D/Headers/module.modulemap" <<'MAP'
module TensorFlowLiteC {
  umbrella header "TensorFlowLiteC/TensorFlowLiteC.h"
  export *
  module * { export * }
  link "c++"
  link "dl"
  link "m"
  link "pthread"
}
MAP
    ARGS="$ARGS -library $D/libTensorFlowLiteC.a -headers $D/Headers"
done

rm -rf "$OUT"
# shellcheck disable=SC2086
xcodebuild -create-xcframework $ARGS -output "$OUT" >/dev/null
echo "$SHA256" > "$OUT/.sha256"
echo "wrote $(pwd)/$OUT"
