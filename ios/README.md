# SoundWatch iOS (PoC trên iPhone)

App iOS chạy đúng model `assets/models/student_int8.tflite` (DS-CNN int8, cùng file sẽ nạp cho ESP32-S3) trên âm thanh
micro iPhone, theo pipeline của bản Python: 16 kHz, cửa sổ 1 s, hop 0.5 s, gate −55 dBFS, log-mel 49×40,
PostProcessor (k-of-n, cooldown, DANGER lặp tới khi xác nhận). Cảnh báo bằng icon, nhãn tiếng Việt, màu theo mức, rung
Core Haptics. Khi khoá màn hình thì gửi notification.

## Cấu trúc

```
ios/
  project.yml                 XcodeGen (target iOS 17, SwiftUI)
  Config.xcconfig             bundle id / team mặc định; ghi đè bằng Local.xcconfig (không commit)
  Local.xcconfig.example
  Packages/LiteRTC/           Swift package bọc thư viện C LiteRT (TensorFlowLiteC.xcframework do script tạo)
  scripts/
    export_ios_assets.py      sinh SoundWatch/Resources từ repo Python
    fetch_litert.sh           tải LiteRT C chính thức, kiểm SHA-256, đóng gói lại xcframework tĩnh
  SoundWatch/
    App/        SoundWatchApp, AppState (ObservableObject)
    Core/       SoundWatchConfig, RingBuffer, FeatureExtractor (Accelerate), TFLiteClassifier,
                PostProcessor, Engine, WavReader
    Audio/      AudioCapture (AVAudioEngine → 16 kHz), DemoFilePlayer
    Services/   Haptics (Core Haptics), Notifier (local notification)
    UI/         ContentView: mặt đồng hồ, cảnh báo, lịch sử, cài đặt, debug
    Resources/  student_int8.tflite, soundwatch_config.json, demo_scenario.wav,
                golden_features.json, demo_expected_events.json (2 file cuối chỉ cho test)
  SoundWatchTests/            XCTest: golden feature, classifier, postprocessor, end-to-end
```

## TFLite / LiteRT

Dùng **thư viện C của LiteRT 2.3.0** (`TensorFlowLiteC.xcframework`, chính là binary mà Swift package chính thức
`google-ai-edge/LiteRT` tag v2.3.2 tải về, cùng URL và checksum), gọi C API từ Swift, delegate XNNPACK như interpreter
Python. Không dùng thẳng package chính thức vì:

- SPM phải clone cả repo LiteRT (~460 MB) và đòi Git LFS nằm trong toolchain của Xcode;
- Xcode nhúng framework tĩnh của package như framework động rồi báo lỗi thiếu `Info.plist`;
- wrapper Swift `TensorFlowLite` của package không có kiểu tensor `int8`, nên không chạy được model full int8.

`scripts/fetch_litert.sh` tải zip chính thức, kiểm SHA-256, rồi đóng gói lại cùng file `.a` thành xcframework thư viện
tĩnh. `xcodegen generate` tự chạy script này (`preGenCommand`). Không dùng CocoaPods, không convert sang Core ML.

## Build lại từ đầu

Cần Mac có Xcode 15+ (đã thử Xcode 27.1), Homebrew, [uv](https://docs.astral.sh/uv/).

```bash
brew install xcodegen
cd soundwatch
uv run ios/scripts/export_ios_assets.py   # chỉ cần khi đổi model/config; Resources đã commit sẵn
cd ios
cp Local.xcconfig.example Local.xcconfig  # sửa SW_BUNDLE_ID (và SW_TEAM_ID nếu biết)
xcodegen generate                         # tự tải LiteRT lần đầu (~7 MB)
open SoundWatch.xcodeproj
```

`uv sync` của repo không chạy trên macOS vì `tensorflow-cpu` không có wheel cho Mac. Script export khai báo dependency
riêng (PEP 723, dùng `ai-edge-litert` 2.3.0) nên chạy được bằng `uv run` như trên. Trên Linux vẫn chạy được bằng
`uv sync && uv run python ios/scripts/export_ios_assets.py`.

Không cần CocoaPods và không có `.xcworkspace`: mở thẳng `SoundWatch.xcodeproj`.

## Cài lên iPhone (Apple ID miễn phí)

1. iPhone: Settings → Privacy & Security → **Developer Mode** → bật, khởi động lại.
2. Cắm iPhone vào Mac, bấm Trust trên iPhone.
3. Xcode → Settings → Accounts → thêm Apple ID (tạo ra "Personal Team").
4. Chọn target SoundWatch → Signing & Capabilities → Team = Personal Team. Nếu bundle id bị trùng thì đổi
   `SW_BUNDLE_ID` trong `Local.xcconfig` rồi chạy lại `xcodegen generate`.
5. Chọn iPhone ở thanh trên, scheme SoundWatch → Product → Scheme → Edit Scheme → Run → Build Configuration =
   **Release** (đo thời gian đúng hơn Debug) → Run.
6. Lần đầu iPhone báo "Untrusted Developer": Settings → General → VPN & Device Management → Apple ID của bạn → Trust.
7. Mở app, cho phép **Micro** và **Thông báo**.

Profile của Personal Team hết hạn sau 7 ngày: trước buổi báo cáo phải build và cài lại.

### Bằng dòng lệnh

```bash
xcodebuild -project SoundWatch.xcodeproj -scheme SoundWatch -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath build -allowProvisioningUpdates \
  DEVELOPMENT_TEAM=<TEAM_ID> build
xcrun devicectl list devices
xcrun devicectl device install app --device <DEVICE_ID> build/Build/Products/Release-iphoneos/SoundWatch.app
xcrun devicectl device process launch --device <DEVICE_ID> <SW_BUNDLE_ID>
```

`-allowProvisioningUpdates` cần Apple ID đã đăng nhập trong Xcode ít nhất một lần.

## Test

```bash
xcodebuild test -project SoundWatch.xcodeproj -scheme SoundWatch \
  -destination 'platform=iOS Simulator,name=iPhone 15'
```

Đổi `name=` theo simulator có trên máy (`xcrun simctl list devices available`). Simulator chỉ chạy được trên Mac Apple
silicon (LiteRT không có lát x86_64 cho simulator).

| Test | Kiểm |
|---|---|
| `FeatureExtractorTests` | log-mel so với 5 golden case của `export-golden`, max \|diff\| < 1e-3; silence = log(1e-6); RingBuffer; thời gian feature + suy luận < 20 ms |
| `ClassifierTests` | input golden case 4 đã chuẩn hoá → xác suất khớp Python `TFLiteClassifier` trong ±1/256; SF Symbols tồn tại |
| `PostProcessorTests` | port `tests/test_wp5_postprocess.py` (tham số cũ của `tests/_legacy_cfg.py`) + `bg_margin` |
| `EndToEndTests` | Engine offline trên `demo_scenario.wav`: sự kiện không lặp khớp `demo_expected_events.json` (cùng lớp, lệch ≤ 0.5 s) |

Kết quả (Xcode 27.1, simulator iOS 27.1, Mac Apple silicon):

| Mục | Kết quả |
|---|---|
| Golden log-mel, max \|diff\| (silence / sine / noise / chirp / sum_of_sines) | 0 / 9.5e-7 / 4.8e-7 / 9.5e-7 / 9.5e-7 |
| Classifier golden case 4 | trùng từng giá trị với Python |
| Demo 60 s: xác suất từng hop (119 hop) | lệch tối đa 0, cùng các hop bị gate |
| Demo 60 s: sự kiện | 12/12 trùng Python (6 sự kiện mới + 6 lần nhắc DANGER), cùng thời điểm |
| Feature + suy luận / hop, Release | 0.20 ms + 0.63 ms (Debug: 5.7 + 1.3 ms) |

## Dùng app

- Màn hình chính: mặt đồng hồ tròn. Lúc nghe: giờ, "Đang nghe…", vòng cung mức âm lượng (−80…0 dBFS).
- Cảnh báo: nền màu theo mức (đỏ DANGER, cam WARN, xanh INFO), icon, nhãn, độ tin cậy, giờ phát hiện. DANGER nhấp nháy
  2 Hz và giữ tới khi chạm "Nhấn để xác nhận" (ack gửi về PostProcessor, dừng nhắc lại). WARN/INFO tự tắt sau 6 s.
- Rung theo mẫu `vibration` của từng lớp; DANGER cường độ 1.0, WARN 0.7, INFO 0.5. DANGER lặp lại thì rung lại nhưng
  không thêm dòng lịch sử.
- **Lịch sử**: 10 sự kiện gần nhất. **Cài đặt**: bật/tắt từng lớp (lưu lại giữa các lần mở).
- **Debug**: xác suất 8 lớp với vạch ngưỡng, RMS dBFS, cờ gated, thời gian feature/suy luận (và max mỗi hop), nút
  "Phát file demo" (đưa `demo_scenario.wav` vào Engine theo thời gian thực, tuỳ chọn phát ra loa). Khi phát demo, micro
  tạm dừng; hết file thì quay lại micro.
- Mở app với launch argument `-demo` (Xcode: Edit Scheme → Run → Arguments) thì phát file demo ngay và hiện Debug.
- Nền / khoá màn hình: `UIBackgroundModes: audio` giữ micro chạy; cảnh báo DANGER và WARN gửi notification (DANGER có
  nút "Xác nhận"), kèm rung hệ thống.
