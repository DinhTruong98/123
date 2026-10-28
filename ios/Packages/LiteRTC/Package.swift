// swift-tools-version: 5.9
// LiteRT (TensorFlow Lite) C library, prebuilt by Google: the binary of the official LiteRT Swift
// package (github.com/google-ai-edge/LiteRT, tag v2.3.2), repackaged by ios/scripts/fetch_litert.sh
// as a static library XCFramework. Run that script once before building.
//
// Not the official package directly because (1) resolving it clones the ~460 MB LiteRT repo and
// needs Git LFS inside Xcode's toolchain, (2) Xcode embeds its static framework and then fails
// ("did not contain an Info.plist"), and (3) its Swift `TensorFlowLite` wrapper has no int8 tensor
// type, so it cannot run a fully int8 model. The app calls the C API (`import TensorFlowLiteC`).
import PackageDescription

let package = Package(
    name: "LiteRTC",
    platforms: [.iOS(.v15)],
    products: [
        .library(name: "TensorFlowLiteC", targets: ["TensorFlowLiteC"]),
    ],
    targets: [
        .binaryTarget(name: "TensorFlowLiteC", path: "TensorFlowLiteC.xcframework"),
    ]
)
