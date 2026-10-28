# SoundWatch PoC – Implementation Plan

> Đồng hồ nhận diện âm thanh môi trường cho người khiếm thính — bản Proof-of-Concept chạy **hoàn toàn trên máy tính** (PC/laptop), được thiết kế để sau này port sang vi điều khiển (ESP32-S3 / nRF52840) với TFLite Micro.
>
> Tài liệu này dành cho các coding agent. Mỗi Work Package (WP) có phạm vi, interface, file cần tạo và tiêu chí nghiệm thu (acceptance criteria) rõ ràng. Agent chỉ làm đúng WP được giao, không đổi interface chung nếu chưa cập nhật mục 4 của tài liệu này.

---

## 0. Tóm tắt

PoC gồm 4 khối chạy trên PC:

1. **Training pipeline** – tải dataset công khai, gom về 8 lớp mục tiêu, trích đặc trưng log-mel, train mô hình teacher (YAMNet) và student (DS-CNN nhỏ), distill, lượng tử hoá int8 → file `.tflite`.
2. **Real-time engine** – đọc âm thanh từ micro laptop (hoặc file WAV), ring buffer 1 s, suy luận bằng **model int8 `.tflite`** (đúng model sẽ nạp xuống MCU), làm mượt xác suất, sinh sự kiện cảnh báo.
3. **Watch simulator UI** – cửa sổ pygame mô phỏng mặt đồng hồ tròn 240×240: icon to + nhãn tiếng Việt + màu theo mức nguy hiểm + mô phỏng rung + lịch sử sự kiện + panel debug (xác suất từng lớp, spectrogram).
4. **Evaluation & benchmark** – đánh giá offline (F1, confusion matrix), đánh giá streaming (tỉ lệ báo nhầm/giờ, độ trễ), đo kích thước model, RAM ước tính, thời gian suy luận; sinh báo cáo.

**Ngoài phạm vi (non-goals):** firmware MCU, BLE, app điện thoại, cloud. Nhưng mọi tham số (feature, model, icon) phải xuất được ra dạng dùng lại cho firmware (mục WP4, WP6).

---

## 1. Yêu cầu chức năng

| ID | Yêu cầu |
|---|---|
| F1 | Nhận diện 7 lớp âm thanh cảnh báo + 1 lớp nền (`background`) từ cửa sổ âm thanh 1 giây, 16 kHz mono. |
| F2 | Chạy real-time từ micro máy tính; cập nhật dự đoán mỗi 0.5 s. |
| F3 | Chạy được từ file WAV (real-time hoặc nhanh nhất có thể) để test/demo không cần micro. |
| F4 | Khi phát hiện âm cảnh báo: hiện icon + nhãn + màu + độ tin cậy trên mặt đồng hồ mô phỏng, chạy mẫu rung tương ứng (mô phỏng bằng hiệu ứng rung khung + thanh pattern). |
| F5 | Lớp nguy hiểm (`danger`) nhấp nháy đỏ và lặp cảnh báo đến khi người dùng xác nhận (phím Space / click). |
| F6 | Màn hình lịch sử: 10 sự kiện gần nhất (icon, nhãn, giờ). |
| F7 | Màn hình cài đặt: bật/tắt cảnh báo từng lớp. |
| F8 | Chế độ headless (không màn hình, không micro) để agent/CI chạy test: `SDL_VIDEODRIVER=dummy`, nguồn âm thanh là file. |
| F9 | Mọi tham số (lớp, ngưỡng, icon, màu, mẫu rung, tham số feature) nằm trong một file config YAML duy nhất. |

## 2. Yêu cầu phi chức năng / mục tiêu định lượng

Phần cứng đích đã được nâng lên **ESP32-S3 có PSRAM** (quyết định của chủ dự án, sau WP3); ngân sách
model/RAM cũ cho nRF52840 được giữ lại để tham khảo.

| Chỉ số | Mục tiêu PoC (ESP32-S3 + PSRAM) | nRF52840 – cũ |
|---|---|---|
| Macro-F1 trên test split (student int8) | ≥ 0.80 | |
| Recall các lớp `danger` (student int8) | ≥ 0.85 | |
| Sụt giảm accuracy khi float32 → int8 | ≤ 2 điểm % | |
| Kích thước model `.tflite` int8 | ≤ 500 KB | ≤ 100 KB |
| Tensor arena ước tính | ≤ 512 KB | ≤ 150 KB |
| MACs / lần suy luận | ≤ 50 M | ≤ 10 M |
| Báo nhầm trên kịch bản chỉ có âm nền | ≤ 2 lần / giờ |
| Độ trễ từ lúc âm bắt đầu → hiện cảnh báo (median) | ≤ 1.5 s |
| Thời gian feature + suy luận trên PC | ≤ 20 ms / cửa sổ |
| UI | ≥ 30 FPS, cảnh báo hiện ≤ 100 ms sau khi engine phát sự kiện |

---

## 3. Công nghệ

- **Python 3.11**, quản lý bằng `uv` (hoặc pip + `requirements.txt`).
- **TensorFlow 2.x + Keras** (chọn TF vì xuất `.tflite` int8 cho TFLite Micro thuận nhất). WP0 phải chọn và **pin** một version mà cả 3 việc sau đều chạy: (a) load YAMNet từ `tensorflow_hub` / Kaggle Models, (b) train Keras model, (c) convert full-int8 TFLite. Nếu Keras 3 gây lỗi với hub thì dùng `tf_keras`. Interpreter: `tf.lite.Interpreter` hoặc `ai-edge-litert`.
- `numpy`, `scipy`, `soundfile`, `librosa` (chỉ để test đối chiếu feature, KHÔNG dùng trong engine), `sounddevice` (micro), `pygame` (UI), `pyyaml`, `pandas`, `scikit-learn` (metrics), `matplotlib` (biểu đồ), `pytest`, `tqdm`.
- Font tiếng Việt: **Be Vietnam Pro** hoặc **Noto Sans** (SIL OFL) — bundle file `.ttf` trong `assets/fonts/` kèm file license.
- Icon: **Google Material Symbols** (Apache 2.0) hoặc **Tabler Icons** (MIT) — PNG 128 px, bundle trong `assets/icons/` kèm license.

### 3.1 Quyết định môi trường (đã chốt ở WP0)

- Python **3.11** (uv), **tensorflow-cpu 2.21.0 + keras 3.15.1** — đã kiểm tra: train Keras, convert full-int8 TFLite, chạy `tf.lite.Interpreter` đều OK. Version pin trong `pyproject.toml` / `uv.lock`; agent không tự thêm dependency mà không ghi chú.
- **YAMNet**: không dùng `tensorflow_hub` (tfhub.dev bị chặn trong môi trường dev). Tải trọng số `https://storage.googleapis.com/audioset/yamnet.h5` + code model (`yamnet.py`, `params.py`, `features.py`) từ `https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/` vào `artifacts/cache/yamnet/`.
- **ESC-50**: tải bằng `git clone --depth 1 https://github.com/karolpiczak/ESC-50` (zip qua codeload bị chặn). **UrbanSound8K**: Zenodo record 1203745. **FSD50K**: chỉ tải chọn lọc (HTTP Range trên split-zip, ~170 MB) các clip baby_cry / smoke_alarm, lọc theo nhãn + tag Freesound + điểm YAMNet (`download --with-fsd50k`, `build-dataset --with-fsd50k`). Thêm: smoke_alarm 340/67/90, baby_cry 604/101/195 cửa sổ (train/val/test). FSD50K rất ít chuông báo khói thật nên ESC-50 `clock_alarm` vẫn được giữ làm smoke_alarm.
- `sounddevice` import lazy trong `MicSource` (container dev không có PortAudio; trên máy thật thì có).
- Mọi CLI command là một module có `add_arguments(parser)` và `main(args) -> int`, đăng ký trong `src/soundwatch/__main__.py`. Module mỗi WP đã có stub; agent chỉ thay stub của mình.
- Đường dẫn chuẩn: `src/soundwatch/paths.py`. Config: `soundwatch.config.load_config()`.

---

## 4. Kiến trúc & interface chung (CONTRACT – không tự ý thay đổi)

### 4.1 Luồng dữ liệu

```
AudioSource (mic | file | scenario)
   │  block float32 [-1,1], 16 kHz mono, kích thước tuỳ ý
   ▼
RingBuffer (16000 mẫu) ── mỗi hop 8000 mẫu (0.5 s) lấy 1 cửa sổ 1 s
   ▼
EnergyGate (RMS dBFS < gate_db → bỏ qua suy luận, coi như background)
   ▼
compute_logmel(window) → float32 (49, 40)
   ▼
Classifier.predict(features) → probs float32 (num_classes,)
   ▼
PostProcessor.update(probs, t) → list[AlertEvent]
   ▼
queue.Queue ──► UI (pygame main thread)
             └► EventLogger (CSV/JSONL)
```

Engine chạy trong **worker thread**; UI chạy trong **main thread**; giao tiếp chỉ qua `queue.Queue` chứa các message ở 4.3.

### 4.2 Tham số âm thanh & feature (cố định, firmware sẽ dùng y hệt)

| Tham số | Giá trị |
|---|---|
| sample_rate | 16000 |
| window | 1.0 s = 16000 mẫu |
| hop (giữa 2 lần suy luận) | 0.5 s = 8000 mẫu |
| frame_length / n_fft | 512 (32 ms), cửa sổ Hann |
| frame_step | 320 (20 ms) |
| số frame | 49 = 1 + (16000 − 512) // 320 (không padding) |
| n_mels | 40, fmin 20 Hz, fmax 8000 Hz, mel scale HTK |
| công thức | `power = |rfft(frame * hann)|^2` → `mel = power @ mel_fb` → `logmel = log(mel + 1e-6)` |
| chuẩn hoá | `(logmel − mean) / std` với mean, std **scalar** tính trên tập train, lưu trong `artifacts/models/feature_stats.json` |
| shape input model | (49, 40, 1) |

Feature phải viết bằng **numpy thuần** (`rfft`, ma trận mel tự dựng) để dễ port sang C/CMSIS-DSP. Không dùng `tf.signal` hay librosa trong engine.

### 4.3 Kiểu dữ liệu & interface Python

```python
# src/soundwatch/types.py
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

class Level(IntEnum):
    INFO = 0
    WARN = 1
    DANGER = 2

@dataclass(frozen=True)
class SoundClass:
    id: int
    key: str            # "smoke_alarm"
    label_vi: str       # "Báo cháy"
    level: Level
    icon: str           # tên file trong assets/icons, vd "fire.png"
    color: tuple[int, int, int]
    vibration: list[int]  # ms bật/tắt xen kẽ, bắt đầu bằng bật
    threshold: float
    cooldown_s: float
    enabled: bool = True

@dataclass(frozen=True)
class AlertEvent:
    class_id: int
    confidence: float
    t_audio: float      # thời điểm (giây) tính theo luồng audio
    t_wall: float       # time.time()

@dataclass(frozen=True)
class StatusUpdate:     # gửi mỗi hop, cho panel debug
    t_audio: float
    rms_dbfs: float
    probs: np.ndarray   # (num_classes,)
    gated: bool
    infer_ms: float

@dataclass(frozen=True)
class AckCommand:       # UI → engine khi người dùng xác nhận
    class_id: int

@dataclass(frozen=True)
class SettingsCommand:  # UI → engine bật/tắt lớp
    class_id: int
    enabled: bool
```

```python
# src/soundwatch/audio/features.py
def mel_filterbank(sr=16000, n_fft=512, n_mels=40, fmin=20.0, fmax=8000.0) -> np.ndarray: ...  # (257, 40)
def compute_logmel(window: np.ndarray) -> np.ndarray: ...  # (16000,) float32 → (49, 40) float32, CHƯA chuẩn hoá
def normalize(logmel: np.ndarray, stats: dict) -> np.ndarray: ...

# src/soundwatch/audio/sources.py
class AudioSource(Protocol):
    sample_rate: int
    def blocks(self) -> Iterator[np.ndarray]: ...   # float32 1-D
    def close(self) -> None: ...
class MicSource: ...        # sounddevice.InputStream, blocksize 1600
class FileSource: ...       # (path, realtime: bool); tự resample về 16 kHz mono
class ArraySource: ...      # từ np.ndarray, phục vụ test

# src/soundwatch/engine/classifier.py
class Classifier(Protocol):
    num_classes: int
    def predict(self, features: np.ndarray) -> np.ndarray: ...  # (49,40) đã chuẩn hoá → probs
class TFLiteClassifier: ...  # tự quantize input / dequantize output theo scale/zero_point của model
class KerasClassifier: ...   # để so sánh float
class DummyClassifier: ...   # trả xác suất theo kịch bản, dùng khi model chưa có

# src/soundwatch/engine/postprocess.py
class PostProcessor:
    def __init__(self, classes: list[SoundClass], smooth_n: int = 3, k_of_n: tuple[int, int] = (2, 3)): ...
    def update(self, probs: np.ndarray, t_audio: float) -> list[AlertEvent]: ...
    def ack(self, class_id: int) -> None: ...
    def set_enabled(self, class_id: int, enabled: bool) -> None: ...
```

**Logic `PostProcessor` (bắt buộc):**
1. Trung bình trượt `smooth_n` vector xác suất gần nhất.
2. Lớp `c` (≠ background) "kích hoạt" nếu xác suất đã làm mượt ≥ `threshold[c]` ở ít nhất `k` trong `n` hop gần nhất.
3. Mỗi lớp có `cooldown_s`; trong cooldown không phát lại sự kiện.
4. Lớp `DANGER` sau khi phát: phát lại mỗi `danger_repeat_s` (mặc định 5 s) cho tới khi nhận `ack`, hoặc âm đó biến mất liên tục 10 s.
5. Lớp bị tắt (`enabled=False`) không bao giờ phát.
6. (WP8, tuỳ chọn, mặc định tắt) `postprocess.bg_margin`: nếu đặt, lớp `c` chỉ "trên ngưỡng" khi thêm
   điều kiện `smoothed[c] − smoothed[background] ≥ bg_margin`. Trường `PostprocessConfig.bg_margin: float | None = None`
   — tương thích ngược (config cũ không có trường này vẫn chạy y như trước).
7. (Luật hai tầng cho DANGER, tuỳ chọn) `postprocess.danger_fast_threshold`: nếu đặt, lớp `DANGER` đang bật, ngoài
   cooldown và không có occurrence đang active sẽ báo **ngay ở một hop** có xác suất đã làm mượt ≥ giá trị này
   (và thoả `bg_margin` nếu bật), không cần đủ k-of-n. Lịch sử hit, cooldown, lặp, ack, clear giữ nguyên; WARN/INFO
   không đổi. `PostprocessConfig.danger_fast_threshold: float | None = None` (tắt). Config hiện dùng 255/256 =
   0.99609375 (output int8 bão hoà). Số liệu trước/sau: [danger_fast_path.md](danger_fast_path.md).

### 4.4 Config (`config/soundwatch.yaml`)

```yaml
audio:
  sample_rate: 16000
  window_s: 1.0
  hop_s: 0.5
  n_fft: 512
  frame_step: 320
  n_mels: 40
  fmin: 20
  fmax: 8000
  gate_dbfs: -55
postprocess:
  smooth_n: 3            # WP8 đã chỉnh trên val: smooth_n 1, k_of_n [2, 2] (xem config/soundwatch.yaml)
  k_of_n: [2, 3]
  danger_repeat_s: 5
  # bg_margin: 0.2       # tuỳ chọn (WP8), mặc định tắt
  danger_fast_threshold: 0.99609375   # luật hai tầng cho DANGER (mục 4.3 quy tắc 7); null = tắt
model:
  tflite_path: artifacts/models/student_int8.tflite
  stats_path: artifacts/models/feature_stats.json
classes:            # thứ tự = chỉ số output của model; background LUÔN ở cuối
  - {key: smoke_alarm,   label_vi: "Báo cháy",     level: 2, icon: fire.png,     color: [230, 40, 40],  vibration: [800,200,800,200,800], threshold: 0.60, cooldown_s: 3}
  - {key: car_horn,      label_vi: "Còi xe",       level: 2, icon: car.png,      color: [230, 40, 40],  vibration: [150,100,150,100,150], threshold: 0.60, cooldown_s: 3}
  - {key: siren,         label_vi: "Còi cấp cứu",  level: 2, icon: siren.png,    color: [230, 40, 40],  vibration: [600,150,200,150,600], threshold: 0.60, cooldown_s: 3}
  - {key: baby_cry,      label_vi: "Em bé khóc",   level: 1, icon: baby.png,     color: [255, 140, 0],  vibration: [500,250,500],         threshold: 0.70, cooldown_s: 8}
  - {key: door_knock,    label_vi: "Gõ cửa / Chuông cửa", level: 1, icon: door.png, color: [255, 140, 0], vibration: [150,150,150], threshold: 0.70, cooldown_s: 8}
  - {key: dog_bark,      label_vi: "Chó sủa",      level: 0, icon: dog.png,      color: [40, 140, 255], vibration: [200],                 threshold: 0.75, cooldown_s: 15}
  - {key: water_running, label_vi: "Nước chảy",    level: 0, icon: water.png,    color: [40, 140, 255], vibration: [600],                 threshold: 0.75, cooldown_s: 30}
  - {key: background,    label_vi: "Yên tĩnh",     level: 0, icon: ear.png,      color: [120, 120, 120], vibration: [],                   threshold: 1.01, cooldown_s: 0}
```

---

## 5. Cấu trúc repo

```
soundwatch-poc/
├── pyproject.toml                # hoặc requirements.txt, đã pin version
├── Makefile                      # các lệnh ở mục 7
├── README.md
├── config/soundwatch.yaml
├── assets/
│   ├── icons/   (*.png + LICENSE)
│   └── fonts/   (*.ttf + LICENSE)
├── src/soundwatch/
│   ├── __init__.py
│   ├── __main__.py               # CLI: python -m soundwatch <subcommand>
│   ├── config.py                 # load YAML → dataclass, validate
│   ├── types.py
│   ├── audio/  features.py  sources.py  ringbuffer.py
│   ├── data/   download.py  label_map.py  build_dataset.py  augment.py  scenarios.py
│   ├── models/ dscnn.py  teacher.py  train.py  distill.py  quantize.py  export_c.py
│   ├── engine/ classifier.py  postprocess.py  pipeline.py  logger.py
│   ├── ui/     app.py  screens.py  widgets.py  theme.py
│   ├── eval/   offline.py  streaming.py  benchmark.py  report.py
│   └── tools/  export_icons.py  export_golden.py
├── tests/
└── artifacts/                    # .gitignore: data/, models/, reports/, logs/
```

---

## 6. Work Packages

Thứ tự phụ thuộc:

```
WP0 ──┬── WP1 (data) ──────┐
      ├── WP2 (features) ──┼── WP3 (train) ── WP4 (quantize/export) ──┐
      ├── WP5 (engine) ────┤                                          ├── WP7 (eval) ── WP8 (tích hợp, demo)
      └── WP6 (UI) ────────┘──────────────────────────────────────────┘
```

Sau WP0, các WP1, WP2, WP5, WP6 chạy **song song**. WP5 dùng `DummyClassifier` và WP6 dùng event giả cho tới khi có model.

---

### WP0 – Scaffolding (làm trước, 1 agent)

**Việc cần làm**
- Tạo cấu trúc repo mục 5, `pyproject.toml` với version đã pin, `.gitignore`, `Makefile`, `README.md` khung.
- `types.py`, `config.py` (load + validate: background ở cuối, threshold trong khoảng (0, 1.01], vibration có số phần tử lẻ hoặc rỗng, icon tồn tại).
- CLI khung `python -m soundwatch {download,build-dataset,train-teacher,train-student,quantize,export,eval,make-scenarios,eval-streaming,benchmark,run,report}` — mỗi lệnh tạm in "not implemented".
- Kiểm tra môi trường: script `python -m soundwatch doctor` in version TF, có load được YAMNet không, có micro không, pygame có chạy headless không.
- Cấu hình `pytest`, `ruff`.

**Nghiệm thu:** `make install && make test` chạy xanh; `python -m soundwatch doctor` chạy; `config/soundwatch.yaml` load và validate được.

---

### WP1 – Data pipeline

**Dataset**
| Dataset | Nguồn | License | Dùng cho |
|---|---|---|---|
| ESC-50 | github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 | Nhiều lớp + background |
| UrbanSound8K | Zenodo / thư viện `soundata` | CC BY-NC 3.0 | car_horn, siren, dog_bark, background |
| FSD50K (tuỳ chọn, bật bằng `--with-fsd50k`) | Zenodo | CC (theo từng clip) | smoke_alarm, doorbell, knock, baby cry, water |

**Ánh xạ nhãn (`label_map.py`)** — agent phải kiểm tra lại tên lớp trong file metadata gốc (`esc50.csv`, `UrbanSound8K.csv`, FSD50K `vocabulary.csv`) trước khi code:

| Lớp đích | ESC-50 | UrbanSound8K | FSD50K (kiểm tra tên chính xác) |
|---|---|---|---|
| smoke_alarm | `clock_alarm` (tạm dùng làm proxy) | – | Smoke detector / Fire alarm / Alarm |
| car_horn | `car_horn` | `car_horn` | Vehicle horn, car horn, honking |
| siren | `siren` | `siren` | Siren |
| baby_cry | `crying_baby` | – | Baby cry / Crying, sobbing |
| door_knock | `door_wood_knock` | – | Knock, Doorbell |
| dog_bark | `dog` | `dog_bark` | Bark |
| water_running | `pouring_water`, `water_drops`, `toilet_flush` | – | Water tap, faucet / Pour |
| background | tất cả lớp ESC-50 còn lại | `air_conditioner`, `children_playing`, `drilling`, `engine_idling`, `jackhammer`, `street_music` | (không dùng) |

Ghi chú: loại `gun_shot` (US8K) và `cat` (ESC-50, dễ lẫn với em bé) — giữ `cat` trong background **có chủ đích** để model học phân biệt.

**Xử lý**
1. `download.py`: tải + giải nén vào `artifacts/data/raw/`, kiểm tra checksum, idempotent (chạy lại không tải lại).
2. Resample về 16 kHz mono float32.
3. **Chia split theo fold gốc để tránh rò rỉ**: ESC-50 fold 1–3 train, 4 val, 5 test; US8K fold 1–8 train, 9 val, 10 test; FSD50K dùng dev/eval gốc (tách 10% dev làm val).
4. Cắt cửa sổ 1 s, hop 0.5 s. Với clip thuộc lớp cảnh báo: **bỏ cửa sổ im lặng** (RMS thấp hơn RMS lớn nhất của clip quá 20 dB), vì clip ESC-50 5 s thường có đoạn lặng nhưng vẫn mang nhãn. Clip ngắn hơn 1 s thì pad 0 ở cuối.
5. Cân bằng: ghi số mẫu từng lớp; background có thể undersample tối đa 3× lớp lớn nhất còn lại.
6. Lưu `artifacts/data/windows/{train,val,test}.npz` gồm `wave (N,16000) float16`, `label (N,) int`, `source_clip (N,) str`.
7. Tách riêng một thư mục **noise bank** (`artifacts/data/noise/`) từ các clip background của **train split** để dùng cho augment và tạo kịch bản.
8. `augment.py` (chỉ áp dụng khi train, online): trộn noise SNR ngẫu nhiên 0–20 dB, gain ±6 dB, time-shift ±0.2 s, (tuỳ chọn) pitch shift ±2 semitone, SpecAugment trên log-mel (2 time mask ≤ 8 frame, 2 freq mask ≤ 6 bin).

**Chế độ `--quick`**: chỉ ESC-50, để agent khác test pipeline trong < 5 phút.

**Nghiệm thu:** `make data` tạo 3 file npz; script in bảng số mẫu mỗi lớp mỗi split; test xác nhận không có `source_clip` nào xuất hiện ở 2 split; mọi lớp có ≥ 1 mẫu ở test.

---

### WP2 – Feature extraction

**Việc cần làm**
- `features.py` đúng spec mục 4.2, numpy thuần, vector hoá (không vòng lặp Python theo frame).
- `normalize` + hàm tính `feature_stats.json` từ train split.
- `tools/export_golden.py`: xuất 5 cửa sổ mẫu (`input.wav`, `logmel.npy`, và `.h` mảng C) vào `artifacts/golden/` để sau này kiểm tra bản C trên MCU cho ra kết quả trùng.
- (Tuỳ chọn) `features_fixed.py`: bản mô phỏng fixed-point int16 để ước lượng sai số khi chạy trên MCU không có FPU.

**Test**
- Shape (49, 40), dtype float32, không NaN với input toàn 0.
- So với `librosa.feature.melspectrogram(..., center=False, htk=True, power=2.0)` sai số tuyệt đối trên log-mel < 1e-3.
- Sine 1 kHz → bin mel có năng lượng lớn nhất ứng với ~1 kHz.
- Thời gian < 2 ms / cửa sổ trên laptop.

**Nghiệm thu:** test xanh; có golden vectors.

---

### WP3 – Model & training

**Teacher (`teacher.py`)**
- YAMNet pretrained (AudioSet), input waveform 16 kHz. Lấy embedding 1024 chiều (trung bình theo frame) cho mỗi cửa sổ 1 s → train head `Dense(256, relu) → Dropout(0.3) → Dense(8, softmax)`.
- Cache embedding vào `artifacts/cache/yamnet_{split}.npz` để không tính lại.
- Lưu logits teacher trên train split (không augment) cho distillation.

**Student (`dscnn.py`)** – DS-CNN, input (49, 40, 1):
```
Conv2D(64, (10,4), stride (2,2), same) → BN → ReLU
4 × [DepthwiseConv2D(3×3, same) → BN → ReLU → Conv2D(64, 1×1) → BN → ReLU]
GlobalAveragePooling2D → Dropout(0.3) → Dense(8) → Softmax
```
- Có tham số `width` (32/64) và `n_blocks` (3/4) để làm bảng so sánh S/M.
- Chỉ dùng op mà TFLite Micro hỗ trợ (Conv2D, DepthwiseConv2D, AveragePool/Mean, FullyConnected, Softmax, ReLU). BN phải fold được khi convert.

**Training (`train.py`, `distill.py`)**
- Adam lr 1e-3, cosine decay, batch 128, 60 epoch, early stopping theo val macro-F1 (patience 10).
- Class weight nghịch tần suất (hoặc focal loss) để tăng recall lớp hiếm.
- Distillation: `loss = α·CE(y, s) + (1−α)·T²·KL(softmax(t/T) ‖ softmax(s/T))`, α = 0.5, T = 4. Teacher logits lấy từ cache (cùng cửa sổ trước augment — chấp nhận lệch nhỏ, ghi chú trong report).
- Seed cố định; log ra `artifacts/logs/<run>/` (CSV metrics từng epoch + config đã dùng).
- Lưu `artifacts/models/student_float.keras`, `teacher_head.keras`.

**Nghiệm thu:** teacher val macro-F1 ≥ 0.85; student float (có distill) val macro-F1 ≥ 0.80; có bảng so sánh student không distill vs có distill.

---

### WP4 – Quantization & export

**Việc cần làm**
- `quantize.py`: TFLiteConverter, `optimizations=[DEFAULT]`, representative dataset = 500 cửa sổ train (đã chuẩn hoá), `supported_ops=[TFLITE_BUILTINS_INT8]`, `inference_input_type=int8`, `inference_output_type=int8` → `student_int8.tflite`. Cũng xuất `student_float.tflite` để so sánh.
- (Tuỳ chọn) Quantization-Aware Training nếu sụt > 2 điểm.
- `export_c.py`: xuất `artifacts/firmware/model_data.cc/.h` (mảng `alignas(16) const unsigned char g_model[]`), `feature_params.h` (toàn bộ hằng số mục 4.2 + mean/std + scale/zero-point input), `labels.h` (tên lớp, level, mẫu rung).
- In danh sách op trong model + kích thước + ước lượng tensor arena (dùng `tf.lite.experimental.Analyzer` hoặc tính từ tensor lớn nhất đồng thời).

**Nghiệm thu:** `TFLiteClassifier` với model int8 chạy trong engine; chênh accuracy float vs int8 trên test ≤ 2 điểm; file `.tflite` ≤ 100 KB; chỉ chứa op được TFLite Micro hỗ trợ.

---

### WP5 – Real-time engine

**Việc cần làm**
- `ringbuffer.py`: buffer cố định 16000 mẫu float32, không cấp phát lại bộ nhớ khi chạy (giống firmware).
- `sources.py`: `MicSource`, `FileSource(realtime=True|False)`, `ArraySource`.
- `classifier.py`: `TFLiteClassifier`, `KerasClassifier`, `DummyClassifier`.
- `postprocess.py`: đúng logic mục 4.3.
- `pipeline.py`: class `Engine(config, source, classifier, out_queue, in_queue)` chạy trong thread, mỗi hop: gate → feature → predict → postprocess → đẩy `StatusUpdate` + `AlertEvent`; đọc `AckCommand`/`SettingsCommand` từ `in_queue`. Dừng sạch khi source hết hoặc gọi `stop()`.
- `logger.py`: ghi mọi `AlertEvent` ra `artifacts/logs/events_<timestamp>.jsonl`.
- CLI: `python -m soundwatch run --source mic|<file.wav> [--headless] [--model path] [--fast]`. `--headless` in sự kiện ra console thay vì mở UI.
- Đo và log: `infer_ms`, `feature_ms`, tỉ lệ cửa sổ bị gate bỏ qua.

**Test**
- Ring buffer: ghi các block kích thước ngẫu nhiên, cửa sổ lấy ra khớp với slice tương ứng của mảng gốc.
- PostProcessor: chuỗi probs tổng hợp → đúng số sự kiện; cooldown; danger lặp lại; ack dừng lặp; lớp bị tắt không phát.
- Pipeline với `ArraySource` + `DummyClassifier` chạy headless, kết thúc, sinh đúng sự kiện mong đợi.

**Nghiệm thu:** chạy `run --source sample.wav --headless --fast` in đúng sự kiện; chạy với micro thật in sự kiện khi phát âm thanh mẫu từ điện thoại.

---

### WP6 – Watch simulator UI

**Layout cửa sổ (pygame, mặc định 960×600)**
```
┌──────────────────────────┬─────────────────────────────┐
│                          │ DEBUG PANEL                 │
│     ╭──────────────╮     │ • Thanh xác suất 8 lớp      │
│    │  MẶT ĐỒNG HỒ  │    │   (vạch ngưỡng từng lớp)    │
│    │  tròn 240×240 │    │ • Spectrogram log-mel cuộn  │
│    │  scale ×2     │    │ • RMS dBFS, gated?          │
│     ╰──────────────╯     │ • infer_ms, FPS             │
│  [mẫu rung đang chạy]    │ • Log sự kiện gần nhất      │
└──────────────────────────┴─────────────────────────────┘
```
Có cờ `--no-debug` chỉ hiện mặt đồng hồ (dùng khi demo cho người dùng).

**Các màn hình trên mặt đồng hồ** (vẽ trên surface 240×240, cắt mask tròn, rồi scale):
1. **Idle**: giờ hiện tại lớn, chữ "Đang nghe…", vòng cung quanh viền thể hiện mức âm lượng (RMS).
2. **Alert**: nền màu theo level (đỏ/cam/xanh, tối hơn ~30%), icon 96×96 màu trắng ở giữa, nhãn tiếng Việt có dấu cỡ lớn, thanh độ tin cậy dạng cung, giờ phát hiện. `DANGER`: nhấp nháy nền 2 Hz + lặp tới khi ack. Lớp khác tự về Idle sau 6 s.
3. **History**: danh sách 10 sự kiện (icon nhỏ 24 px + nhãn + HH:MM:SS), cuộn bằng phím mũi tên / con lăn.
4. **Settings**: danh sách lớp với công tắc bật/tắt → gửi `SettingsCommand`.

**Mô phỏng rung**: khi có sự kiện, chạy pattern `vibration` của lớp: trong thời gian "bật", mặt đồng hồ dịch ngẫu nhiên ±3 px mỗi frame và hiện biểu tượng rung; bên dưới vẽ thanh timeline của pattern. Phải chạy bất đồng bộ theo thời gian (không dùng `sleep`), UI không bị đứng.

**Điều khiển**: `Space`/click vào mặt = xác nhận (ack); `H` = History; `S` = Settings; `Esc` = về Idle; `Q` = thoát.

**Assets**: `theme.py` load icon từ `assets/icons/` theo config, tô lại màu trắng; font Be Vietnam Pro.

**`tools/export_icons.py`**: từ cùng icon PNG xuất `artifacts/firmware/icons.h` gồm (a) bitmap 1-bit 48×48 cho OLED SSD1306 (định dạng XBM cho U8g2) và (b) RGB565 96×96 cho TFT GC9A01. Thêm ảnh preview `icons_preview.png`.

**Chế độ phát triển độc lập**: `python -m soundwatch.ui.app --demo` sinh event giả theo vòng lặp (mỗi lớp lần lượt) để phát triển UI khi chưa có engine/model.

**Test (headless, `SDL_VIDEODRIVER=dummy`)**
- Render mỗi màn hình không lỗi, chụp ra `artifacts/reports/ui_<screen>.png` (dùng làm hình cho báo cáo).
- Nhận `AlertEvent` → chuyển sang màn hình Alert trong ≤ 1 frame.
- Ack trên DANGER → gửi `AckCommand` vào queue.
- Nhãn tiếng Việt có dấu render đúng (không ra ô vuông).

**Nghiệm thu:** `--demo` chạy mượt ≥ 30 FPS; có 4 ảnh chụp màn hình; có `icons.h`.

---

### WP7 – Evaluation & benchmark

**Offline (`eval/offline.py`)** cho 4 model: teacher, student float, student float (không distill), student int8:
- accuracy, macro-F1, precision/recall/F1 từng lớp, confusion matrix (PNG), **recall riêng cho các lớp DANGER**.

**Streaming (`data/scenarios.py` + `eval/streaming.py`)**
- Sinh 20 kịch bản × 3 phút từ **test split**: nền liên tục từ noise bank (test) + chèn ngẫu nhiên 3–6 sự kiện ở SNR {0, 5, 10, 20} dB, kèm file nhãn `scenario_XX.csv` (class, onset, offset).
- Sinh thêm 10 kịch bản × 6 phút **chỉ có nền** để đo báo nhầm.
- Chạy engine `--fast` trên từng kịch bản, so sự kiện với nhãn: một sự kiện đúng nếu cùng lớp và nằm trong [onset, offset + 2 s].
- Chỉ số: event recall từng lớp, false alarms/giờ, độ trễ (alert − onset) median và p90, kết quả theo SNR.
- Thêm mục quét ngưỡng (`threshold` 0.4–0.9) → vẽ đường đánh đổi recall vs báo nhầm/giờ, đề xuất ngưỡng.

**Benchmark (`eval/benchmark.py`)**
- Số tham số, MACs (tính tay từ kiến trúc), kích thước `.tflite` float vs int8, tensor arena ước tính.
- Thời gian feature và suy luận trên PC (median trên 1000 lần).
- Tỉ lệ cửa sổ bị energy gate bỏ qua trong kịch bản chỉ có nền (để ước lượng tiết kiệm pin).
- Bảng **ước tính** cho MCU: MACs / (tốc độ MAC dự kiến của Cortex-M4F 64 MHz với CMSIS-NN, và ESP32-S3 240 MHz với ESP-NN) — ghi rõ là ước tính.

**Report (`eval/report.py`)**: gom tất cả vào `artifacts/reports/report.md` + hình PNG.

**Nghiệm thu:** `make eval` sinh `report.md` đầy đủ các bảng/hình trên; đối chiếu với mục tiêu mục 2 (đánh dấu đạt/chưa đạt).

---

### WP8 – Tích hợp & demo

- Nối model int8 thật vào engine + UI; chạy end-to-end với micro.
- `make demo` = `python -m soundwatch run --source mic`; `make demo-file` chạy kịch bản mẫu.
- Chỉnh ngưỡng trong config theo kết quả WP7.
- Script quay demo: chạy kịch bản có đủ 7 lớp, quay màn hình (ghi lại các frame pygame ra MP4/GIF bằng `imageio`, tuỳ chọn).
- (Tuỳ chọn) `record` subcommand: thu âm môi trường thật bằng micro, gán nhãn, lưu vào `artifacts/data/custom/` để fine-tune — giảm lệch miền (domain shift) giữa dataset và micro thật.
- Hoàn thiện `README.md`: cài đặt, tải dữ liệu, train, chạy demo, cấu trúc code, ánh xạ module PC → firmware (mục 9).

**Nghiệm thu:** người mới clone repo làm theo README chạy được demo bằng model có sẵn trong ≤ 15 phút (không tính thời gian train).

---

## 7. Lệnh (Makefile)

| Lệnh | Việc |
|---|---|
| `make install` | cài dependency |
| `make test` | pytest (headless) |
| `make data` / `make data-quick` | WP1 |
| `make train` | teacher + student + distill |
| `make quantize` | int8 + export firmware headers + icons |
| `make eval` | offline + streaming + phân tích báo nhầm + benchmark + report (đo FPS UI) |
| `make tune` | (WP8) sinh kịch bản **val** + quét hậu xử lý → `artifacts/reports/tuning.md` |
| `make demo` / `make demo-file` | chạy UI với mic / file |
| `make demo-headless` / `make demo-gif` | demo không UI (console) / quay GIF cửa sổ UI |
| `make record CLASS=… SECONDS=…` | (WP8) thu âm micro → `artifacts/data/custom/<lớp>/` |
| `make all` | data → train → quantize → eval |

---

## 8. Quy tắc cho coding agent

1. Chỉ sửa file thuộc WP của mình; muốn đổi interface mục 4 phải cập nhật tài liệu này trong cùng commit và ghi rõ lý do.
2. Mọi thứ phải chạy được **headless** và không cần micro (dùng `FileSource`/`ArraySource`, `SDL_VIDEODRIVER=dummy`).
3. Không commit dataset, model, cache (đặt trong `artifacts/`, đã gitignore). Ngoại lệ: một `student_int8.tflite` nhỏ + `feature_stats.json` trong `assets/models/` để demo không cần train lại.
4. Không hardcode tham số — đọc từ config.
5. Seed cố định cho mọi thứ có ngẫu nhiên.
6. Mỗi WP kèm test pytest; `make test` phải chạy < 2 phút và không cần mạng (test dùng dữ liệu tổng hợp / sine / noise).
7. Code có type hints, chạy `ruff` sạch.
8. Khi xong WP: cập nhật mục "Trạng thái" ở cuối tài liệu này và ghi kết quả đo được.

---

## 9. Ánh xạ PC → firmware (để port sau PoC)

| Module PC | Tương ứng trên MCU |
|---|---|
| `MicSource` | Driver PDM/I2S + DMA (nRF52 PDM, ESP32-S3 I2S) |
| `RingBuffer` | Buffer vòng int16 tĩnh 16000 mẫu (32 KB) |
| `compute_logmel` | CMSIS-DSP `arm_rfft_fast_f32` + mel filterbank thưa; kiểm tra bằng golden vectors |
| `TFLiteClassifier` | TFLite Micro + CMSIS-NN / ESP-NN, `model_data.cc` |
| `PostProcessor` | Port nguyên logic sang C (vài chục dòng) |
| `EnergyGate` | Tính RMS trên DMA block, cho phép CPU ngủ khi yên tĩnh |
| UI pygame | U8g2 (OLED) hoặc LVGL (TFT tròn), `icons.h` |
| Mô phỏng rung | Motor ERM/LRA qua DRV2605 hoặc GPIO + transistor |
| Thread + queue | FreeRTOS task + queue (ESP32-S3) / timer + ISR (nRF52) |

---

## 10. Rủi ro & cách xử lý

| Rủi ro | Xử lý |
|---|---|
| Thiếu dữ liệu báo cháy/khói thật (ESC-50 không có) | Bật FSD50K; bổ sung clip từ Freesound (CC0/CC-BY); thu âm chuông báo khói thật (nút Test); tạm dùng `clock_alarm` làm proxy và ghi rõ trong báo cáo. |
| Lệch miền giữa dataset và micro laptop | Augment noise mạnh; lệnh `record` thu dữ liệu thật để fine-tune; đánh giá riêng trên dữ liệu thu thật. |
| Nhầm `siren` vs `car_horn`, `baby_cry` vs `cat` | Xem confusion matrix; tăng dữ liệu; có thể gộp lớp nếu không cải thiện được. |
| Nhiều âm cùng lúc (multi-label) | PoC dùng softmax single-label; ghi là hạn chế, hướng phát triển: sigmoid multi-label. |
| Version TF / hub / TFLite không tương thích | WP0 kiểm tra và pin; fallback: dùng embedding YAMNet đã cache. |
| Báo nhầm nhiều | Quét ngưỡng (WP7), k-of-n, cooldown, energy gate, tăng dữ liệu background. |

---

## 11. Mốc thời gian gợi ý

| Tuần | Nội dung |
|---|---|
| 1 | WP0; bắt đầu song song WP1, WP2, WP5 (dummy), WP6 (demo) |
| 2 | Xong WP1, WP2; WP3 teacher |
| 3 | WP3 student + distill; WP5, WP6 hoàn thiện |
| 4 | WP4; tích hợp model thật vào engine + UI |
| 5 | WP7 đánh giá, chỉnh ngưỡng |
| 6 | WP8 demo, README, báo cáo; (tuỳ chọn) thu dữ liệu thật và fine-tune |

---

## 12. Trạng thái

| WP | Agent | Trạng thái | Ghi chú / kết quả |
|---|---|---|---|
| WP0 | agent WP0 | Xong | uv + Python 3.11, tensorflow-cpu 2.21 / keras 3.15 (pin), CLI khung, `doctor`, config validate (mục 3.1). |
| WP1 | agent WP1 | Xong | ESC-50 + UrbanSound8K + FSD50K chọn lọc (baby_cry, smoke_alarm); split theo fold, không rò rỉ clip; noise bank train/test. |
| WP2 | agent WP2 | Xong | log-mel numpy thuần, sai số so với librosa < 1e-3; ~0.15 ms/cửa sổ; golden vectors + `.h`. |
| WP3 | agent WP3 | Xong | Teacher YAMNet+head: test macro-F1 0.823. Student DS-CNN (width 96, 4 block, 24.5 M MACs) có distill: val macro-F1 0.828 / test 0.795; không distill: val 0.796 / test 0.715. |
| WP4 | agent WP4 | Xong | int8 208 KB, arena ~117 KB, chỉ op TFLM; sụt float→int8 +0.08 điểm % accuracy (trùng top-1 98.9 %); `model_data.cc`, `feature_params.h`, `labels.h`. |
| WP5 | agent WP5 | Xong | Engine thread + queue, ring buffer tĩnh, energy gate, PostProcessor (mục 4.3), `run` (mic/file, headless, fast). |
| WP6 | agent WP6 | Xong | UI pygame 960×600, 4 màn hình, mô phỏng rung, debug panel; `icons.h` (XBM 48 px + RGB565 96 px); 30 FPS (khả năng ~160 FPS). |
| WP7 | agent WP7 | Xong | 20 kịch bản sự kiện + 10 × 6 phút nền (test); hậu xử lý cũ (smooth 3, k/n 2/3): recall 0.46, 67 báo nhầm/giờ, trễ median 2.0 s; quét ngưỡng; benchmark; `report.md`. |
| WP8 | agent WP8 | Xong | `tune` trên kịch bản **val** → smooth_n 1, k_of_n 2/2, ngưỡng 0.90–0.99; test: recall 0.40 (DANGER TB 0.54), **23 báo nhầm/giờ**, trễ median **1.43 s** (p90 3.2 s). `fa-analysis`: chỉ ~6 % báo nhầm do nhiễu nhãn nền. `record`, `--record-gif`, `--ui-stats`, README. |

---

## 13. Kết quả & hạn chế (sau WP8)

**Số liệu cuối** (student int8 `assets/models/student_int8.tflite`; chi tiết `artifacts/reports/report.md`):

| Hạng mục | Kết quả | Mục tiêu |
|---|---|---|
| Offline test macro-F1 / accuracy | 0.795 / 0.792 (teacher 0.823) | ≥ 0.80 ❌ (sát) |
| Recall DANGER offline (smoke / car_horn / siren) | 0.90 / 0.73 / **0.56** | ≥ 0.85 ❌ |
| Sụt float → int8 | +0.08 điểm % | ≤ 2 ✅ |
| Kích thước / arena / MACs | 208 KB / ~117 KB / 24.5 M | ≤ 500 KB / 512 KB / 50 M ✅ |
| Streaming test: event recall (tổng / DANGER TB) | 0.40 / 0.54 (SNR 20 dB: 0.74, 0 dB: 0.21) | – |
| Báo nhầm / giờ trên nền (test) | 23 (val: 10) | ≤ 2 ❌ |
| Độ trễ median / p90 | 1.43 s / 3.2 s (oracle: 1.22 s) | ≤ 1.5 s ✅ |
| PC: feature + suy luận | ~0.5 ms / cửa sổ | ≤ 20 ms ✅ |
| UI | 30 FPS (giới hạn), khả năng ~160 FPS | ≥ 30 ✅ |
| Ước tính ESP32-S3 240 MHz + ESP-NN | ~51–102 ms / suy luận (10–20 % CPU mỗi hop) | – |

**Tinh chỉnh hậu xử lý (WP8):** với ràng buộc trễ median ≤ 1.5 s, chỉ còn smooth_n = 1 (hoặc smooth 2 + k/n 1/1);
trên val không cấu hình nào đạt ≤ 2 hoặc ≤ 5 báo nhầm/giờ, nên chọn ngân sách 10/giờ (smooth_n 1, k_of_n 2/2).
Nới trễ lên ~1.8 s thì đạt ≤ 2/giờ trên val (smooth 2, k/n 2/2) nhưng recall giảm (0.43). Model int8 quá tự tin
(xác suất bão hoà 0.996) nên ngưỡng gần như không còn tác dụng; quy tắc `bg_margin` không giúp gì.

**Hạn chế đã biết**
- `siren` yếu nhất (recall offline 0.56) và là nguồn báo nhầm lớn nhất: model nhầm âm có cao độ ổn định / hài âm
  (nhạc đường phố – violin, máy khoan, điều hoà, động cơ) thành còi. Phân tích bằng YAMNet (`fa-analysis`) cho thấy
  phần lớn **không** phải nhiễu nhãn (nền thật sự không có còi/chó) → lỗi model; hướng xử lý: hard-negative mining,
  mixup nền–sự kiện, hiệu chỉnh nhiệt độ (temperature scaling) trước khi đặt ngưỡng.
- `smoke_alarm` một phần vẫn là proxy ESC-50 `clock_alarm` (FSD50K có rất ít chuông báo khói thật); tiếng dế bị nhầm
  thành báo cháy.
- FSD50K chỉ tải chọn lọc (baby_cry, smoke_alarm); các lớp khác chỉ từ ESC-50 / UrbanSound8K (ít clip `door_knock`).
- Kịch bản streaming là âm thanh tổng hợp (nền + sự kiện trộn theo SNR), ~1 giờ nền test → sai số thống kê lớn cho
  báo nhầm/giờ; val (2 giờ) và test cho 10 vs 23 báo nhầm/giờ.
- Lệch miền sang micro thật **chưa được kiểm tra** (container phát triển không có micro); `record` đã có để thu dữ liệu
  thật và fine-tune.
- Single-label softmax: không xử lý được nhiều âm cùng lúc.
