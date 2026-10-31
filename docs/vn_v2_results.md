# SoundWatch VN v2: kết quả

Nhánh `vn-v2`, xây từ `vn-finetune`. Model mới nằm ở `assets/models/vn_v2/` (`student_int8.tflite`, `feature_stats.json`), dùng
với `config/soundwatch_vn_v2.yaml`. Model đang chạy trong app (`assets/models/student_int8.tflite`, gọi là **cu**) và model
VN v1 (`assets/models/vn/`, gọi là **vn1**) **không bị thay**. Báo cáo chi tiết (json/md) ở `docs/vn_v2_reports/`.

## Tóm tắt

- **vn2 = DS-CNN 96→192 softmax, log-mel, distill**, train trên dữ liệu đã cân bằng lại `smoke_alarm`, **hiệu chỉnh nhiệt độ**
  (T = 2.11, đã gộp vào lớp Dense cuối), ngưỡng và luật hai tầng **chọn lại trên val** (`danger_fast_threshold` 0.95 thay
  cho 255/256). Cùng kích thước, cùng op, cùng MACs với cu/vn1. Có thêm **chế độ ngữ cảnh** (`indoor` / `street` / `auto`).
- **Còi VN tăng rõ:**
  - Mức cửa sổ, test: recall còi VN 0.83 (cu) / 0.93 (vn1) → **0.98**; xe cứu hoả 0.69 / 0.90 → **0.98**; xe đi ngang qua 0.75 / 0.92 → **0.96**.
  - Mức sự kiện, kịch bản VN test: còi 0.87 / 0.91 → **0.96**; recall tổng (còi + báo cháy VN) 0.66 / 0.75 → **0.79**; trễ median 1.43 / 1.23 → **1.05 s**.
- **Báo nhầm trên nền không tăng so với cu** ở cả bốn bộ: test 37 → 32, val 22 → 22, VN test 16 → 13, VN val 32 → 16 lần/giờ.
  Ở **chế độ `street`**, nền VN còn **5** (VN test) và **4** (VN val) lần/giờ.
- **Chưa đạt hoàn toàn** tiêu chí "lớp khác trên dữ liệu gốc không tụt quá 2 điểm" ở mức cửa sổ: test smoke_alarm −5.6, door_knock −9.5
  (42 cửa sổ), val door_knock −2.8, water_running −2.6. Ở mức sự kiện trên kịch bản tổng hợp thì smoke_alarm test 0.42 → 0.33 (12 sự kiện,
  tức 1 sự kiện), các lớp còn lại bằng hoặc cao hơn cu.
- Các thử nghiệm **không** cải thiện và **không** được chọn: PCEN (kém hơn rõ), sigmoid đa nhãn + trộn hai cửa sổ (tốt cho lớp WARN/INFO
  nhưng kém ở sự kiện DANGER ngắn), cửa sổ 2 s (tốt nhất theo điểm val chính nhưng kém ở còi ngắn, MACs ×2, cần sửa app). Chi tiết ở mục 4.
- **Đề xuất (mục 10):** dùng vn2 thay vn1 làm ứng viên thử trên iPhone. Port sang app chỉ cần **đổi model + ngưỡng** (cùng front-end
  log-mel, cùng softmax). Chế độ `street` cần thêm UI. Chưa thay model mặc định của app khi chưa thử ngoài đường thật.

## 1. Thay đổi so với VN v1 và lý do

| # | Thay đổi | Lý do | Bật bằng |
|---|---|---|---|
| 1 | **Metadata front-end theo từng model** trong `feature_stats.json` → `params`: `frontend` (`logmel`/`pcen`), `pcen` (hằng số), `output` (`softmax`/`sigmoid`), `window_s` (`audio.features.ModelMeta`). `run`, `Engine`, `eval-vn`, `eval-streaming`, `eval-fast-path`, `tune`, `quantize`, `export`, `export-golden` tự đọc từ file stats của model. Output sigmoid **không** bao giờ bị softmax lại (`classifier.to_probs`; tự nhận ra khi graph kết thúc bằng op LOGISTIC). | Để model PCEN / sigmoid / cửa sổ 2 s chạy được bằng đúng công cụ cũ. Model cũ không có các khoá này nên chạy y như trước. | file stats của model |
| 2 | **Front-end PCEN** (`frontend: pcen`): cùng mel power (40 mel HTK, FFT 512, hop 320), thay `log` bằng PCEN cố định, không học, khởi tạo bộ làm mượt lại ở mỗi cửa sổ, numpy thuần, vector hoá. | PCEN thường bền hơn với thay đổi âm lượng và nền ồn. Bản không trạng thái giữa các cửa sổ để port MCU/iOS cho dễ. | `train-student --frontend pcen` |
| 3 | **Đầu ra sigmoid đa nhãn** (`output: sigmoid`): cùng DS-CNN, head sigmoid, BCE trên 8 output (`background` = "không có âm cảnh báo"). **Trộn hai cửa sổ** (`--p-mix`): cộng thêm một cửa sổ khác trong miền mel power. Cửa sổ thêm vào là lớp sự kiện khác (mức tương đối ±6 dB) hoặc nền (SNR 0–20 dB). Nhãn là hợp của hai nhãn; `background` = 1 chỉ khi không có sự kiện nào. Distill: xác suất softmax của teacher làm nhãn mềm cho các mẫu không trộn. | Âm thật hay chồng lên nhau, vd còi xe + còi cấp cứu ngoài đường. Softmax buộc phải chọn một lớp. | `train-student --output sigmoid [--p-mix 0.5]` |
| 4 | **Hiệu chỉnh nhiệt độ:** fit T trên val bằng NLL (softmax, 1 giá trị) hoặc BCE (sigmoid, 1 giá trị cho mỗi lớp), rồi **gộp vào Dense cuối** (`W/T`, `b/T`) trước khi quantize, nên op TFLite không đổi. Báo cáo ECE / reliability / số cửa sổ bão hoà trước và sau. | Model cũ quá tự tin: khoảng 50% cửa sổ bão hoà ở 255/256. Vì vậy ngưỡng phải đặt 0.99 và luật hai tầng chỉ còn mang nghĩa "bão hoà". | `quantize --calibrate` |
| 5 | **Cân bằng lại `smoke_alarm`:** VN tối đa 2 cửa sổ/clip, **chỉ ở train** (`CUSTOM_MAX_WINDOWS_TRAIN`). | VN chiếm 63% lớp này ở train. Nghi đây là nguyên nhân smoke_alarm trên dữ liệu gốc tụt 4.4 điểm ở vn1. | `build-dataset` (mặc định mới) |
| 6 | **Chế độ ngữ cảnh** (`modes:` trong config, quy tắc 9 trong `engine/postprocess.py`). `indoor`: như cũ. `street`: tắt dog_bark và baby_cry, hạ car_horn xuống INFO (không luật hai tầng, không lặp). `auto`: chuyển sang `street` khi trung vị RMS các hop trong 30 s > −35 dBFS (trễ 3 dB). Đổi chế độ lúc chạy qua `ModeCommand` / `PostProcessor.set_mode`, mỗi lần đổi được ghi vào `mode_log`, log và `engine.stats["mode_changes"]`. Có cờ `run --mode` và `eval-streaming/eval-fast-path/tune --mode`. Khi config có `modes:` thì xuất ra `labels.h` (`kModeDisabled`, `kModeLevel`, `SW_AUTO_*`) và JSON cho iOS (`modes`). | Báo nhầm trên nền VN chủ yếu là dog_bark (đường phố có chó, xe máy). Còi xe ngoài đường là chuyện bình thường. | `modes:` + `--mode` |
| 7 | **Cửa sổ 2 s** (thử nghiệm): `build-dataset --windows-dir` tạo bộ cửa sổ thứ hai, không đụng noise/events. Có thêm `train-teacher/train-student/quantize --windows-dir --cache-dir`, `train-teacher --models-dir`, `eval-vn --windows-dir`. | Thêm ngữ cảnh cho còi của xe đi ngang qua. | `window_s` trong stats |

Mọi thay đổi đều **tuỳ chọn và tương thích ngược**. Với `config/soundwatch.yaml` / `config/soundwatch_vn.yaml` và các model cũ, kết
quả giống hệt trước. Đã kiểm:
- 12 sự kiện trong `ios/SoundWatch/Resources/demo_expected_events.json` và xác suất từng hop khớp tuyệt đối.
- `soundwatch_config.json` sinh lại khớp với bản trong `Resources`.
- `feature_params.h`, `labels.h`, `model_data.cc` của model cũ không đổi.
- Số `eval-fast-path` của cu/vn1 trùng với `docs/vn_finetune_results.md`.

`uv run pytest`: 164 → **210** test, `uv run ruff check src tests` sạch. Test mới ở `tests/test_v2_frontend_modes.py` và
`tests/test_v2_train.py`.

### Công thức PCEN (hằng số mặc định)

```
E[t, m] = mel power (bước 1–5 như log-mel), t = 0..48, m = 0..39
T       = time_constant_s · sample_rate / frame_step = 0.4 · 16000 / 320 = 20 frame
s       = (sqrt(1 + 4T²) − 1) / (2T²) = 0.0487656
M[0, m] = E[0, m]                                (khởi tạo lại ở mỗi cửa sổ 1 s)
M[t, m] = (1 − s) · M[t−1, m] + s · E[t, m]
pcen    = (E / (eps + M)^gain + bias)^power − bias^power,   gain 0.98, bias 2, power 0.5, eps 1e-6
x       = (pcen − mean) / std                    (mean/std vô hướng trên train, trong feature_stats.json)
```

Công thức này khớp `librosa.pcen(E.T, sr=16000, hop_length=320, time_constant=0.4, gain=0.98, bias=2, power=0.5, eps=1e-6,
max_size=1, zi=lfilter_zi·E[0])` với sai số < 1e-6 (test `test_pcen_matches_librosa`, ngưỡng 1e-3). Golden vector:
`export-golden --frontend pcen` (hoặc `--stats` của một model PCEN) ghi thêm `case_*/pcen.npy` và `golden_pcen.h`.

## 2. Dữ liệu

- Chỉ build lại cửa sổ từ dữ liệu có sẵn, không tải lại. Lệnh: `build-dataset --with-fsd50k --custom-manifest
  /home/user/vn-sound-data/manifest.csv`, mất 6 phút 45 giây.
- Chỉ lớp `smoke_alarm` ở train thay đổi:

  | | train trước | train sau | VN / tổng |
  |---|---|---|---|
  | smoke_alarm | 916 (VN 576 = 63%) | **532** (VN 192 = **36%**) | 96 clip VN × 2 cửa sổ |

  val/test giữ **đúng từng cửa sổ** như lần trước (5983 / 5990). Mọi bộ kịch bản sinh lại bằng đúng lệnh ở mục 2 của
  `docs/vn_finetune_results.md` (có `--no-demo`) và cho ra cùng file. Số của cu/vn1 trên các bộ này trùng khớp với tài liệu cũ.
- Teacher (YAMNet + head) train lại trên dữ liệu mới: macro-F1 val / test = 0.817 / 0.810 (vn1: 0.814 / 0.799).

## 3. Các lệnh đã chạy

```bash
git checkout -b vn-v2 origin/vn-finetune
P="uv run python -m soundwatch"
$P build-dataset --with-fsd50k --custom-manifest /home/user/vn-sound-data/manifest.csv
# kịch bản: đúng các lệnh make-scenarios ở docs/vn_finetune_results.md mục 2 (test, val, short_{test,val}, vn_{test,val}, --no-demo)
$P train-teacher
# thí nghiệm (2 run song song, mỗi run --threads 2, khoảng 50 phút):
$P train-student --distill --threads 2 --tag v2_sm_lm                                  # A = vn2
$P train-student --distill --threads 2 --output sigmoid --tag v2_sg_lm                 # B
$P train-student --distill --threads 2 --output sigmoid --p-mix 0.5 --tag v2_sg_lm_mix # C
$P train-student --no-distill --threads 2 --output sigmoid --p-mix 0.5 --tag v2_sg_lm_mix_nod  # D
$P train-student --distill --threads 2 --frontend pcen --tag v2_sm_pcen               # E
# F: cửa sổ 2 s (config artifacts/v2_2s/soundwatch_2s.yaml = soundwatch.yaml với window_s: 2.0)
C2="--config artifacts/v2_2s/soundwatch_2s.yaml"; D2="--windows-dir artifacts/v2_2s/windows --cache-dir artifacts/v2_2s/cache"
$P build-dataset $C2 --with-fsd50k --custom-manifest /home/user/vn-sound-data/manifest.csv --windows-dir artifacts/v2_2s/windows
$P train-teacher $C2 $D2 --models-dir artifacts/v2_2s/models
$P train-student $C2 $D2 --distill --tag v2_sm_lm_2s
# quantize + calibrate, KHÔNG copy vào assets (mỗi run một thư mục):
$P quantize --tag <tag> --no-copy --calibrate --out-dir artifacts/models/<tag>   # + $C2 $D2 cho run 2 s
# chọn hậu xử lý trên val (scripts/vn_v2/, xem mục 5):
uv run python scripts/vn_v2/runs_cache.py <tag> config/soundwatch_vn_v2.yaml artifacts/models/<tag>/student_int8.tflite artifacts/models/<tag>/feature_stats.json
uv run python scripts/vn_v2/select_pp.py <tag> config/soundwatch_vn_v2.yaml artifacts/reports/v2/selectF_<tag>.json 22 32 0.05
uv run python scripts/vn_v2/make_cfg.py artifacts/reports/v2/selectF_v2_sm_lm.json config/soundwatch_vn_v2.yaml config/soundwatch_vn_v2.yaml assets/models/vn_v2
# sản phẩm: copy model A vào assets/models/vn_v2/ (feature_stats.json thêm metadata front-end + nhiệt độ)
# đánh giá cuối:
$P eval-vn --split {val,test} --manifest /home/user/vn-sound-data/manifest.csv \
    --model cu assets/models/student_int8.tflite assets/models/feature_stats.json \
    --model vn1 assets/models/vn/student_int8.tflite assets/models/vn/feature_stats.json \
    --model vn2 assets/models/vn_v2/student_int8.tflite assets/models/vn_v2/feature_stats.json
$P eval-fast-path --config config/soundwatch_vn_v2.yaml --values 0.9,0.95,0.97,0.99609375
$P eval-fast-path --config config/soundwatch_vn_v2.yaml --mode street --values 0.95
$P eval-streaming --config config/soundwatch_vn_v2.yaml --no-sweep [--scenarios artifacts/data/scenarios_vn_test] [--mode street]
$P tune --config config/soundwatch_vn_v2.yaml --no-margin --budgets 10,15,22 \
    --thr-grid 0.3,0.35,...,0.95,0.97,0.99,0.99609375          # -> docs/vn_v2_reports/tuning_vn2.md
uv run python scripts/vn_v2/compare.py out.json cu=config/soundwatch.yaml vn1=config/soundwatch_vn.yaml \
    v2_sm_lm=config/soundwatch_vn_v2.yaml v2_sm_lm=config/soundwatch_vn_v2.yaml@street
# firmware (chỉ trong artifacts/, không commit):
$P export --config config/soundwatch_vn_v2.yaml --out artifacts/firmware_vn_v2
$P export-golden --config config/soundwatch_vn_v2.yaml --stats assets/models/vn_v2/feature_stats.json --out artifacts/golden_vn_v2
$P export-golden --frontend pcen --out artifacts/golden_pcen
```

Tổng thời gian train: 3 vòng, mỗi vòng 2 run song song, cộng 2 lần train teacher. Thời gian thực tế khoảng 2 giờ 50 phút.

## 4. Các thí nghiệm

Tất cả dùng DS-CNN 96→192 (công thức mặc định của `train-student`), cùng dữ liệu mới, cùng cách quantize + calibrate. Hậu xử lý
của **mỗi** model (kể cả cu và vn1) được chọn lại bằng **cùng một quy trình trên val** (mục 5), để so model với model chứ không so
ngưỡng với ngưỡng. Cột "điểm val" là recall có trọng số (DANGER ×2) trên val ∪ VN val ∪ DANGER-ngắn-val; đây là tiêu chí chọn cuối.
Mọi số trong bảng này là **val**.

| Lần | Model | int8 macro-F1 val / test (cửa sổ) | Tụt float→int8 (điểm, val / test) | val: recall / DANGER / báo nhầm/giờ | VN val: recall / báo nhầm/giờ | DANGER ngắn val | **Điểm val** |
|---|---|---|---|---|---|---|---|
| | cu (chọn lại ngưỡng) | 0.805 / 0.784 (gốc) | – | 0.515 / 0.596 / 22.0 | 0.811 / 29 | 0.333 | 0.554 |
| | vn1 (chọn lại ngưỡng) | 0.787 / 0.793 (gốc) | – | 0.577 / 0.649 / 19.5 | 0.867 / 5 | 0.313 | 0.584 |
| **A** | **softmax, log-mel, distill = vn2** | **0.786 / 0.779** | +0.05 / −0.05 | 0.554 / 0.632 / 22.0 | 0.867 / 16 | **0.414** | **0.612** |
| B | sigmoid, log-mel, distill, không trộn | 0.780 / 0.775 | +0.47 / −0.55 | 0.531 / 0.649 / 22.0 | 0.856 / 11 | 0.434 | 0.612 |
| C | sigmoid + trộn (p 0.5), distill | 0.750 / 0.763 | −0.02 / +0.38 | 0.577 / 0.614 / 20.0 | 0.867 / 5 | 0.263 | 0.563 |
| D | sigmoid + trộn (p 0.5), **không** distill | 0.753 / 0.762 | −0.21 / +0.09 | 0.585 / 0.667 / 18.5 | 0.867 / 4 | 0.333 | 0.595 |
| E | softmax, **PCEN**, distill | 0.704 / 0.730 | +0.17 / −0.18 | 0.554 / 0.632 / 21.5 | 0.811 / 14 | 0.343 | 0.570 |
| F | softmax, log-mel, distill, **cửa sổ 2 s** | 0.801 / 0.826 (cửa sổ 2 s, không so trực tiếp được) | −0.07 / +0.22 | **0.608 / 0.667 / 20.0** | **0.878 / 5** | 0.364 | **0.614** |

(Macro-F1 của cu/vn1 là của dữ liệu gốc trong `eval-vn`, chỉ để tham khảo. Hàng A–E chấm trên toàn bộ val/test 1 s. Hàng F chấm
trên bộ cửa sổ 2 s.)

Nhận xét theo từng câu hỏi:

- **PCEN vs log-mel (E vs A): PCEN kém hơn rõ.** macro-F1 val 0.704 so với 0.786, smoke_alarm val chỉ 0.52, điểm val 0.570 so với
  0.612. Lý do hợp lý: PCEN với hằng số thời gian 0.4 s là một bộ AGC, nó làm phẳng âm ổn định. Chuông báo cháy là âm ổn định, nên bị
  "nén" đi. Không thử thêm hằng số khác vì hết ngân sách train (một lần thử tốn 50 phút). Code, test và golden vẫn giữ để thử sau
  (`--frontend pcen`, có thể chỉnh `pcen` trong stats).
- **Sigmoid vs softmax (B vs A):** ở mức cửa sổ gần như ngang nhau (0.780 / 0.775 so với 0.786 / 0.779). Sau calibration, sigmoid bắt
  sự kiện DANGER ngắn tốt hơn một chút (0.434 so với 0.414) nhưng báo nhầm VN val cao hơn (11 so với 16, trong khi DANGER val ngang).
  Điểm val 0.612 so với 0.612, tức hoà. Chọn softmax vì app và firmware đang dùng softmax, nên không phải sửa code.
- **Trộn hai cửa sổ (C, D vs B):** làm **tăng** recall các lớp WARN/INFO (val baby_cry, door_knock, water_running) và giảm báo nhầm VN
  (4–5/giờ). Đổi lại, cửa sổ nền bị đoán thành sự kiện nhiều hơn hẳn (test: nền → báo động 0.21 so với 0.13) và sự kiện DANGER ngắn
  **kém đi** (0.26–0.33 so với 0.43). Với p = 0.5, một nửa số cửa sổ nền được cộng thêm sự kiện, nên model ít thấy nền "sạch". Nếu
  thử lại nên giảm p hoặc trộn nền với nền.
- **Distill on/off (C vs D, sigmoid + trộn):** không có xu hướng rõ. macro-F1 val 0.750 so với 0.753, test 0.763 so với 0.762; điểm val
  0.563 so với 0.595 (D nhỉnh hơn, chủ yếu nhờ DANGER ngắn 0.33 so với 0.26). Teacher softmax chỉ dùng được làm nhãn mềm cho mẫu không
  trộn, nên tác dụng bị pha loãng. Với softmax (A) vẫn giữ distill như công thức cũ.
- **Calibration:** xem mục 6. T của softmax là 2.1–2.5 (quá tự tin). Với sigmoid, T theo lớp là 0.44–0.70 (thiếu tự tin, do label
  smoothing 0.1 trong BCE) và gần 0.9–1.0 cho `background`.
- **Giới hạn smoke_alarm (A so với vn1, cùng công thức):** smoke_alarm gốc val 0.940 → **0.970**, test 0.856 → **0.844**. Như vậy cú
  tụt ở test **không** do VN chiếm đa số. Lớp này ở test chỉ có 90 cửa sổ gốc (14 clip); 0.844 so với 0.900 của cu tương đương 5 cửa
  sổ. Smoke VN: val 0.842 → 0.789, test 0.686 → 0.745. Sự kiện smoke VN: VN test 0.59 → 0.63, VN val 0.87 → 0.87.
- **Cửa sổ 2 s (F):** MACs 47.7 M (≤ 50 M), tensor arena khoảng 234 KB, int8 208 KB. Đứng đầu theo recall có trọng số trên val ∪ VN
  val (0.749 so với 0.719) và có báo nhầm VN val thấp (5). Nhưng nó **kém ở sự kiện ngắn**: còi ngắn val 0.18 so với 0.45, vì một tiếng
  còi 0.5–1 s bị pha loãng trong cửa sổ 2 s. Trễ median val cũng cao hơn (1.41 so với 1.27 s). Theo điểm val cuối (có tính DANGER
  ngắn) thì F 0.614 và A 0.612 coi như **hoà**. Mình chọn A vì: (1) còi xe đi ngang qua ngắn là lý do chính của luật hai tầng, (2) app
  iOS và firmware đang dùng cửa sổ 1 s, arena của A chỉ bằng một nửa, (3) A thay vào app chỉ cần đổi file. Để tham khảo (test, chỉ báo
  cáo): F có recall test 0.57 so với 0.49 của A và DANGER 0.72, nhưng báo nhầm VN test 23/giờ, **cao hơn** cu (16). Hướng tiếp theo nên
  ghép hai cửa sổ: dùng 2 s cho âm dài và 1 s cho luật hai tầng.
- **Lần thất bại / sự cố:**
  - Lần build cửa sổ 2 s đầu bị OOM (cgroup khoảng 12 GB, chạy cùng lúc với 2 run train). Đã sửa `build()` để gom cửa sổ theo
    `keep` mà không nối toàn bộ (`_gather_kept`), nên đỉnh RAM giảm khoảng một nửa. Kết quả với 1 s không đổi (có test).
  - Chọn ngưỡng **không có sàn từng lớp** cho A ra cấu hình có điểm val cao hơn, nhưng nó đặt dog_bark ở 255/256: val dog_bark vẫn
    còn, nhưng test dog_bark về 0/13. Vì vậy quy trình cuối thêm sàn "mỗi lớp ≥ recall của cu − 0.05 trên val".
  - **Lỗi trong luật hai tầng** (phát hiện khi chạy demo với vn2). Khi `danger_fast_threshold` < ngưỡng của lớp, cảnh báo qua đường
    nhanh không cập nhật "lần cuối trên ngưỡng". Vì vậy occurrence kết thúc ngay ở hop sau, và lớp đó có thể báo lại một cảnh báo mới
    sau `cooldown_s` thay vì lặp. Mọi config cũ có `danger_fast_threshold` (255/256) ≥ mọi ngưỡng, nên không bị ảnh hưởng (số cu/vn1
    giữ nguyên). Đã sửa: một hop vượt ngưỡng nhanh cũng tính là "còn âm" cho đồng hồ kết thúc (quy tắc 4b). Có test. Mọi bước chọn
    ngưỡng và mọi số trong tài liệu này đã chạy lại sau khi sửa.

## 5. Chọn hậu xử lý (chỉ trên val)

Quy trình (`scripts/vn_v2/select_pp.py`, dùng lại `eval.tune.class_tables`):
1. Suy luận một lần cho mỗi kịch bản val, VN val và DANGER-ngắn-val (`runs_cache.py`). Sau đó chỉ replay `PostProcessor`.
2. Lưới tìm: `danger_fast_threshold` ∈ {tắt, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 255/256} × (smooth, k-of-n) ∈ {(1, 2/2), (1, 1/1),
   (2, 2/3), (1, 2/3)}. Ngưỡng từng lớp lấy trên lưới 0.30–0.95, 0.97, 0.99, 255/256 bằng thuật toán tham lam (DANGER ×2), với ngân
   sách báo nhầm 10–35/giờ trên nền của val ∪ VN val. Điểm xuất phát của mỗi lớp là ngưỡng cao nhất còn giữ được recall ≥ recall của cu
   (config gốc) − 0.05 trên val ∪ VN val.
3. Điều kiện chấp nhận: báo nhầm/giờ trên val ≤ 22 (bằng cu), VN val ≤ 32 (bằng cu), trễ median val ≤ 1.5 s. Chọn cấu hình có
   recall có trọng số lớn nhất.
4. Chọn giữa các model bằng **điểm val** (recall có trọng số trên val ∪ VN val ∪ DANGER-ngắn-val, bảng mục 4).

Cấu hình chọn cho vn2 (`config/soundwatch_vn_v2.yaml`):

| | cu (`soundwatch.yaml`) | vn1 (`soundwatch_vn.yaml`) | **vn2 (`soundwatch_vn_v2.yaml`)** |
|---|---|---|---|
| smooth_n / k_of_n | 1 / 2-of-2 | 1 / 2-of-2 | 1 / **2-of-3** |
| danger_fast_threshold | 255/256 | 255/256 | **0.95** |
| Ngưỡng smoke / car_horn / siren | 0.90 / 0.97 / 0.99 | 0.90 / 0.97 / 0.95 | **0.70 / 0.80 / 0.95** (*) |
| Ngưỡng baby / door / dog / water | 0.99 / 0.99 / 0.99 / 0.99 | như cu | **0.85 / 0.30 / 0.95 / 0.90** |

(*) Quy trình chọn ra siren = 255/256; khi đó siren thực tế chỉ báo qua luật hai tầng (một hop ≥ 0.95). Đặt 0.95 cho kết quả **giống
hệt** trên cả ba bộ val. Mình chọn 0.95 vì với giá trị này siren vẫn báo được khi tắt luật hai tầng.

Quét `danger_fast_threshold` cho vn2 (các ngưỡng khác giữ nguyên, `eval-fast-path`, số val dùng để chọn):

| fast | val: recall / siren / báo nhầm/giờ | VN val: recall / siren / báo nhầm/giờ | (test: recall / báo nhầm/giờ) | (VN test: recall / báo nhầm/giờ) |
|---|---|---|---|---|
| tắt | 0.538 / 0.68 / 18.5 | 0.844 / 0.87 / 14 | 0.465 / 27 | 0.758 / 13 |
| 0.90 | 0.554 / 0.74 / **26.5** | 0.878 / 0.89 / 18 | 0.488 / 40 | 0.791 / 20 |
| **0.95** | **0.554 / 0.74 / 22** | **0.867 / 0.87 / 16** | 0.488 / 32 | 0.791 / 13 |
| 0.97 | 0.546 / 0.68 / 21.5 | 0.867 / 0.87 / 15 | 0.477 / 30 | 0.780 / 13 |
| 255/256 (giá trị cũ) | 0.546 / 0.68 / 19 | 0.844 / 0.87 / 14 | 0.465 / 27 | 0.758 / 13 |

Mức 0.90 có recall cao hơn nhưng vượt ngân sách báo nhầm val (26.5 > 22). Mức 0.95 cho siren val 0.68 → 0.74 so với khi tắt luật hai
tầng hoặc giữ 255/256, đổi lại khoảng +3.5 lần báo nhầm/giờ. `tune` (bản gốc, chỉ trên val, `tuning_vn2.md`) cho kết quả
cùng xu hướng: ngân sách 22/giờ chọn smooth 1, k 1/1, và đặt dog_bark/baby_cry ở 255/256. Quy trình ở trên đã loại cách đặt đó vì có sàn
từng lớp.

## 6. Calibration

Nhiệt độ fit trên val (float), gộp vào Dense cuối. Số đo trên **int8** (mức cửa sổ).

| Model | T | ECE top-label val (trước → sau) | ECE từng lớp val | Cửa sổ bão hoà ≥ 255/256, val (trước → sau) | test (trước → sau) |
|---|---|---|---|---|---|
| cu (không calibrate) | – | 0.069 | 0.021 | 3304 / 5983 (55%) | 2968 / 5990 |
| vn1 (không calibrate) | – | 0.062 | 0.020 | 3158 / 5983 (53%) | 2841 / 5990 |
| **vn2 (A)** | **2.11** | 0.072 → **0.025** | 0.021 → 0.019 | 3748 → **1636** (27%) | 3473 → 1381 |
| E (PCEN) | 2.30 | 0.099 → 0.033 | 0.031 → 0.028 | 3317 → 1074 | 3351 → 853 |
| F (2 s) | 2.48 | 0.063 → 0.028 | 0.018 → 0.012 | 3860 → 1211 / 5064 | 3556 → 1133 / 5059 |
| B (sigmoid) | 0.55–0.70, bg 0.89 | 0.052 → 0.041 | 0.035 → **0.013** | 599 → 1134 | 357 → 956 |
| C (sigmoid + trộn) | 0.57–0.65, bg 1.04 | 0.044 → 0.073 | 0.061 → 0.034 | 392 → 1222 | 432 → 1067 |

NLL float val của vn2: 0.510 → 0.368. Với sigmoid, nhiệt độ được tối ưu cho từng lớp (BCE). Vì vậy ECE từng lớp giảm mạnh, nhưng ECE
top-label có thể tăng. Số cửa sổ bão hoà cũng tăng, vì head sigmoid có label smoothing thiếu tự tin và T < 1 làm nó tự tin hơn.

Reliability của vn2 trên val, int8 (khoảng độ tin cậy của lớp có xác suất cao nhất: số cửa sổ, độ tin cậy TB → độ chính xác):

| Khoảng | Trước | Sau |
|---|---|---|
| 0.47–0.53 | 59: 0.50 → 0.29 | 168: 0.50 → 0.52 |
| 0.60–0.67 | 129: 0.63 → 0.54 | 264: 0.63 → 0.68 |
| 0.73–0.80 | 153: 0.77 → 0.67 | 401: 0.77 → 0.79 |
| 0.87–0.93 | 335: 0.90 → 0.69 | 665: 0.90 → 0.89 |
| 0.93–1.00 | 4936: 0.99 → 0.93 | 3319: 0.98 → 0.97 |

Trước calibration, 83% cửa sổ dồn vào khoảng cuối và độ chính xác thấp hơn độ tin cậy ở mọi khoảng. Sau calibration, độ chính xác
bám sát độ tin cậy. Hệ quả: ngưỡng không còn dồn ở 0.99 (vn2 dùng 0.30–0.95, mục 5), và luật hai tầng dùng 0.95 thay cho "bão hoà".

## 7. So sánh cu / vn1 / vn2

### 7.1 Mức cửa sổ 1 s (`eval-vn`, cùng cửa sổ)

| Chỉ số | val: cu / vn1 / vn2 | test: cu / vn1 / vn2 |
|---|---|---|
| **VN:** macro-F1 | 0.661 / 0.686 / **0.707** | 0.670 / 0.690 / **0.707** |
| **VN:** recall siren | 0.860 / 0.872 / **0.936** | 0.828 / 0.931 / **0.983** |
| **VN:** recall smoke_alarm | 0.561 / 0.842 / 0.789 | 0.664 / 0.686 / **0.745** |
| VN: nền → báo động / nền → siren | 0.095 / 0.079 / 0.089 · 0.011 / 0.015 / 0.012 | 0.124 / 0.118 / 0.138 · 0.006 / 0.006 / 0.015 |
| **Gốc:** macro-F1 | 0.805 / 0.787 / 0.798 | 0.784 / 0.793 / 0.768 |
| Gốc: smoke_alarm / car_horn / siren | 0.866 / 0.773 / 0.962 → **0.970 / 0.789 / 0.968** | 0.900 / 0.733 / 0.561 → **0.844** / 0.775 / 0.580 |
| Gốc: baby_cry / door_knock | 0.851 / 0.771 → 0.881 / **0.743** | 0.815 / 0.857 → 0.821 / **0.762** |
| Gốc: dog_bark / water_running | 0.754 / 0.727 → 0.762 / **0.701** | 0.779 / 0.713 → 0.760 / 0.729 |
| Gốc: background | 0.896 → 0.892 | 0.878 → 0.867 |
| Bão hoà ≥ 255/256 (gốc) | 0.576 / 0.569 / 0.294 | 0.498 / 0.483 / 0.231 |

Số in đậm ở các hàng "Gốc" (cu → vn2) là lớp tụt quá 2 điểm so với cu: smoke_alarm test −5.6 (90 cửa sổ), door_knock test −9.5 (42
cửa sổ, tức 4 cửa sổ), door_knock val −2.8 (35 cửa sổ, 1 cửa sổ), water_running val −2.6. Bảng đầy đủ: `docs/vn_v2_reports/vn_eval_{val,test}.md`.

**Recall còi VN theo nhóm** (cửa sổ, cu / vn1 / vn2):

| Nhóm | val | test |
|---|---|---|
| ambulance | 0.882 / 0.833 / **0.958** | 0.972 / 0.938 / **0.986** |
| fire | 0.867 / 0.908 / **0.942** | 0.688 / 0.902 / **0.983** |
| police | 0.812 / 0.840 / **0.910** | 0.889 / 0.972 / **0.981** |
| other | 0.917 / 0.917 / 0.917 | 0.972 / 0.972 / 0.972 |
| **passing (xe đi ngang qua)** | 0.741 / 0.741 / **0.870** (54) | 0.750 / 0.917 / **0.958** (24) |
| far / mid / near | 0.825 / 0.879 / 0.889 → **0.921 / 0.954 / 0.938** | 0.904 / 0.792 / 0.798 → **0.974 / 0.988 / 0.985** |

### 7.2 Mức sự kiện (mỗi model với config của nó; vn2 street = cùng model + `--mode street`)

| Bộ kịch bản | Recall tổng: cu / vn1 / vn2 / vn2 street | DANGER TB | Báo nhầm/giờ trên nền | Trễ median (s) |
|---|---|---|---|---|
| test (tổng hợp, 86 sự kiện, 1 giờ nền) | 0.42 / 0.44 / **0.49** / 0.38 | 0.56 / 0.53 / 0.56 / 0.53 | 37 / 35 / **32** / 21 | 1.48 / 1.41 / 1.37 / 1.34 |
| val (129 sự kiện, 2 giờ nền) | 0.48 / 0.52 / **0.55** / 0.36 | 0.56 / 0.61 / **0.63** / 0.61 | 22 / 22 / **22** / 12 | 1.36 / 1.23 / 1.27 / 1.27 |
| **VN test** (91 sự kiện, nền VN 1 giờ) | 0.66 / 0.75 / **0.79** / 0.79 | 0.66 / 0.75 / **0.79** / 0.79 | 16 / 20 / **13** / **5** | 1.43 / 1.23 / **1.05** / 1.05 |
| **VN val** (90 sự kiện) | 0.78 / 0.88 / 0.87 / 0.87 | 0.78 / 0.88 / 0.87 / 0.87 | 32 / 5 / 16 / **4** | 1.49 / 1.57 / 1.39 / 1.39 |
| DANGER ngắn test (0.5/1/1.5 s) | 0.34 / 0.33 / 0.31 / 0.30 | – | – | 0.79 / 0.81 / 0.82 / 0.91 |
| DANGER ngắn val | 0.34 / 0.33 / **0.41** / 0.40 | – | – | 0.92 / 0.91 / 0.95 / 0.96 |

Recall từng lớp (cu / vn1 / vn2 / vn2 street):

| Bộ | smoke_alarm | car_horn | siren | baby_cry | door_knock | dog_bark | water_running |
|---|---|---|---|---|---|---|---|
| test | 0.42 / 0.42 / 0.33 / 0.33 | 0.58 / 0.58 / 0.67 / 0.58 | 0.67 / 0.58 / 0.67 / 0.67 | 0.23 / 0.23 / 0.23 / 0 | 0.17 / 0.33 / 0.50 / 0.50 | 0.46 / 0.54 / 0.38 / 0 | 0.42 / 0.42 / 0.67 / 0.67 |
| val | 0.68 / 0.84 / 0.84 / 0.84 | 0.26 / 0.32 / 0.32 / 0.26 | 0.74 / 0.68 / 0.74 / 0.74 | 0.61 / 0.50 / 0.78 / 0 | 0.22 / 0.28 / 0.33 / 0.33 | 0.53 / 0.58 / 0.53 / 0 | 0.33 / 0.39 / 0.33 / 0.33 |
| VN test | 0.46 / 0.59 / **0.63** / 0.63 | – | 0.87 / 0.91 / **0.96** / 0.96 | – | – | – | – |
| VN val | 0.67 / 0.87 / 0.87 / 0.87 | – | 0.89 / 0.89 / 0.87 / 0.87 | – | – | – | – |
| DANGER ngắn test | 0.43 / 0.50 / 0.43 / 0.43 | 0.34 / 0.24 / 0.24 / 0.21 | 0.24 / 0.24 / 0.24 / 0.24 | – | – | – | – |
| DANGER ngắn val | 0.30 / 0.48 / 0.48 / 0.48 | 0.30 / 0.18 / 0.30 / 0.27 | 0.42 / 0.33 / 0.45 / 0.45 | – | – | – | – |

Báo nhầm trên nền theo lớp (số lần; cu / vn1 / vn2 / vn2 street):

| Bộ | giờ nền | smoke_alarm | car_horn | siren | baby_cry | door_knock | dog_bark | water_running |
|---|---|---|---|---|---|---|---|---|
| test | 1 | 7 / 8 / 6 / 6 | 2 / 2 / 2 / 2 | 17 / 13 / **9** / 9 | 3 / 1 / 6 / 0 | 0 / 0 / 1 / 1 | 8 / 10 / 5 / 0 | 0 / 1 / 3 / 3 |
| val | 2 | 3 / 6 / 6 / 6 | 0 / 2 / 3 / 1 | 10 / 10 / 7 / 7 | 12 / 13 / 14 / 0 | 0 / 2 / 6 / 6 | 19 / 11 / 4 / 0 | 0 / 0 / 4 / 4 |
| VN test | 1 | 0 / 2 / 0 / 0 | 4 / 2 / 3 / 4 | 1 / 2 / 1 / 1 | 1 / 0 / 2 / 0 | 0 / 0 / 0 / 0 | 10 / 13 / **7** / 0 | 0 / 1 / 0 / 0 |
| VN val | 1 | 1 / 2 / 1 / 1 | 1 / 1 / 3 / 2 | 3 / 0 / 1 / 1 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 27 / 2 / 11 / 0 | 0 / 0 / 0 / 0 |

Báo nhầm **mức DANGER** (smoke + car_horn + siren; ở street, car_horn là INFO nên không tính): test 26 / 23 / 17 / 15, VN test 5 / 6 /
4 / 1 lần/giờ. vn2 giảm báo nhầm siren trên test tổng hợp (17 → 9). dog_bark vẫn là nguồn báo nhầm chính trên nền VN, và street
loại bỏ toàn bộ số này.

### 7.3 Kích thước, tài nguyên, int8

| | cu | vn1 | vn2 |
|---|---|---|---|
| Kiến trúc | DS-CNN 96→192, 4 khối | giống hệt | giống hệt (softmax, log-mel) |
| int8 / tham số / MACs | 208 KB / 152 k / 24.5 M | 208 KB / 152 k / 24.5 M | 208 KB (213 184 B) / 152 k / 24.5 M |
| Op | CONV_2D, DEPTHWISE_CONV_2D, MEAN, FULLY_CONNECTED, SOFTMAX | như cu | như cu (nhiệt độ đã gộp vào trọng số) |
| Tensor arena (ước tính) | ~117 KB | ~117 KB | ~117 KB |
| Tụt float → int8 (macro-F1, val / test) | – | −0.30 / −0.11 | **+0.05 / −0.05** điểm |
| Thời gian chạy | không đổi | không đổi | không đổi (cùng đồ thị) |

Model 2 s (F): 47.7 M MACs, arena khoảng 234 KB. Model sigmoid: thêm op LOGISTIC, các thông số khác không đổi.

## 8. Chế độ ngữ cảnh

- **`street`** (vn2, `--mode street`). Báo nhầm trên nền VN: VN test 13 → **5**, VN val 16 → **4** lần/giờ (cu: 16 / 32). Trên nền tổng
  hợp: test 32 → 21, val 22 → 12.
- **Cái giá:** baby_cry và dog_bark **không bao giờ báo** (recall về 0, theo thiết kế). car_horn thành INFO nên không có luật hai tầng
  và không lặp: recall car_horn test 0.67 → 0.58, val 0.32 → 0.26, DANGER ngắn car_horn 0.24 → 0.21. siren, smoke_alarm, door_knock,
  water_running không đổi. Tất cả âm còi / báo cháy VN vẫn được bắt như ở chế độ indoor (0.79 / 0.87).
- **`auto`** đã cài và có test (chuyển chế độ theo trung vị RMS, có trễ). Tuy nhiên **chưa đánh giá được** trên kịch bản tổng hợp:
  `make-scenarios` đặt mức nền ngẫu nhiên trong khoảng −45 đến −30 dBFS cho cả nền gốc lẫn nền VN. Trung vị RMS của các kịch bản nền:
  test −44.5 đến −31.1 dBFS, VN test −41.8 đến −30.7 dBFS. Hai phân bố gần như trùng nhau, nên ngưỡng −35 dBFS trong config chỉ là
  **giá trị tạm**. Cần đo với micro thật của đồng hồ/iPhone (trong nhà và ngoài đường) rồi đặt lại. Mặc định của config là `indoor`.
- Mỗi lần đổi chế độ được ghi lại: log `mode <cũ> -> <mới> at t=… (lý do)`, dòng `-- mode … -> …` khi chạy `run --headless`,
  và `engine.stats["mode_changes"]`.

## 9. Cửa sổ 2 s

Đã làm (mục 4, thí nghiệm F): build cửa sổ 2 s riêng (`artifacts/v2_2s/windows`, train 31 650 cửa sổ), train teacher riêng, train
student 99×40, quantize và chọn ngưỡng trên val bằng cùng quy trình. Model chạy được thẳng bằng engine (`window_s: 2.0` trong stats,
ring buffer 32 000 mẫu). Kết luận ở mục 4: hoà với A theo điểm val, tốt hơn ở âm dài, kém ở âm ngắn, tốn gấp đôi tài nguyên. Model
nằm ở `artifacts/models/v2_sm_lm_2s/` (không commit), config chọn ngưỡng ở `docs/vn_v2_reports/select_val_v2_sm_lm_2s.json`.

## 10. Đánh giá theo tiêu chí và đề xuất

| Tiêu chí | vn2 |
|---|---|
| Recall còi VN tăng | ✅ cửa sổ test 0.83 → 0.98 (vn1 0.93), val 0.86 → 0.94. Sự kiện VN test 0.87 → 0.96; ⚠️ VN val 0.89 → 0.87 (−1 sự kiện trên 45). Xe cứu hoả 0.69 → 0.98, xe đi ngang qua 0.75 → 0.96 (test) / 0.74 → 0.87 (val) |
| Lớp khác trên dữ liệu gốc không tụt quá 2 điểm so với cu | ⚠️ mức sự kiện: chỉ smoke_alarm test tụt (0.42 → 0.33 = 1/12 sự kiện), dog_bark test 0.46 → 0.38 (1/13); các lớp khác bằng hoặc cao hơn. ❌ mức cửa sổ: smoke_alarm test −5.6, door_knock test −9.5 / val −2.8, water_running val −2.6 (lớp ít dữ liệu: 35–194 cửa sổ). Cân bằng lại smoke_alarm **không** khắc phục được cú tụt smoke test |
| Báo nhầm/giờ trên nền không tăng so với cu | ✅ test 37 → 32, val 22 → 22, VN test 16 → 13, VN val 32 → 16 |
| Chế độ street: báo nhầm nền VN giảm rõ | ✅ VN test 13 → 5, VN val 16 → 4 |

**Đề xuất:**
1. **Dùng vn2 làm ứng viên thay vn1** cho đợt thử trên iPhone. vn2 tốt hơn vn1 ở còi VN (test; val ngang nhau), DANGER ngắn (val)
   và báo nhầm (trừ VN val: 5 → 16, do dog_bark), và tự tin "đúng mức" hơn hẳn. Port rất rẻ: chỉ đổi `student_int8.tflite` và các ngưỡng / `danger_fast_threshold`
   (mục 11). **Chưa thay model mặc định của app** cho tới khi thử với còi thật ngoài đường, vì tiêu chí "không tụt lớp khác" ở mức cửa
   sổ vẫn chưa đạt (smoke_alarm, door_knock trên test).
2. Thêm nút **Trong nhà / Ngoài đường** vào app trước, chưa cần `auto`. Đây là cách rẻ nhất để cắt báo nhầm ngoài đường (VN: chia 3).
   Chỉ bật `auto` sau khi đo RMS thật.
3. Không theo PCEN và sigmoid + trộn ở dạng hiện tại. Nếu muốn đa nhãn: giảm p trộn (khoảng 0.2), trộn cả nền với nền, bỏ label
   smoothing. Cửa sổ 2 s đáng thử tiếp theo dạng **hai cửa sổ**: 2 s cho k-of-n, 1 s cho luật hai tầng.
4. Dữ liệu vẫn là nút thắt: smoke_alarm gốc ở test chỉ có 14 clip, door_knock 8 clip. Cần thêm clip thật (lệnh `record`), và thêm nền VN
   có chó sủa làm hard negative cho dog_bark.

## 11. Cần sửa gì để chạy vn2 trên iPhone và ESP32 (chưa làm, Swift không đổi)

vn2 dùng **cùng front-end log-mel, cùng softmax, cùng op, cùng kích thước tensor** như model trong app. Nhiệt độ đã nằm trong trọng số.

**App iOS (Swift):**
1. Thay `student_int8.tflite` bằng `assets/models/vn_v2/student_int8.tflite`. Đổi `feature_stats` (mean −2.148, std 4.094). Đổi ngưỡng
   và `danger_fast_threshold: 0.95`, `k_of_n: [2, 3]`. Ngưỡng `danger_fast_threshold` < ngưỡng lớp cần bản sửa quy tắc 4b (mục 4):
   một hop vượt ngưỡng nhanh cũng giữ occurrence. Với ngưỡng của vn2 thì không lớp DANGER nào có ngưỡng > 0.95, nên Swift hiện tại
   vẫn đúng. Có thể sinh `soundwatch_config.json` bằng
   `export_ios_assets.py --config config/soundwatch_vn_v2.yaml --model assets/models/vn_v2/student_int8.tflite --stats
   assets/models/vn_v2/feature_stats.json`. **Lưu ý:** script ghi thẳng vào `ios/SoundWatch/Resources/` nên chỉ chạy khi đã quyết định
   thay model. Phần còn lại không cần sửa code, nếu PostProcessor Swift đọc `k_of_n` từ JSON như hiện tại.
2. **Chế độ ngữ cảnh** (code mới): đọc khoá `modes` trong JSON; thêm nút chọn chế độ. Khi đổi chế độ:
   - tắt các lớp trong `disable`;
   - đổi mức theo `level` (car_horn → INFO: màu xanh, không lặp, không luật hai tầng);
   - xoá occurrence đang chờ ack của lớp bị tắt hoặc bị hạ mức (giống `PostProcessor._apply_mode`).

   `auto`: trung vị RMS (dBFS) của 60 hop gần nhất, so với `street_above_dbfs ± hysteresis_db/2`, chỉ quyết định sau `min_s` giây.
3. Nếu sau này dùng model PCEN / sigmoid / 2 s:
   - JSON có khoá `model_frontend`;
   - PCEN: thay `log(mel + 1e-6)` bằng vòng PCEN ở mục 1 (49×40, hằng số trong `model_frontend.pcen`), `golden_features.json` có trường
     `pcen` để kiểm < 1e-3;
   - sigmoid: **không** softmax / chuẩn hoá lại output;
   - 2 s: ring buffer 32 000 mẫu, 99 frame.

**Firmware ESP32-S3:**
1. `export --config config/soundwatch_vn_v2.yaml` (đã sinh ở `artifacts/firmware_vn_v2/`, không commit) cho `model_data.cc`,
   `feature_params.h` (mean/std mới, k-of-n 2/3) và `labels.h` (ngưỡng mới). Front-end không đổi, nên `golden_features.h` và
   `mel_filterbank.h` không đổi.
2. `labels.h` có thêm `SW_NUM_MODES`, `kModeNames`, `kModeDisabled[mode][class]`, `kModeLevel[mode][class]`, `SW_AUTO_*`.
   PostProcessor C cần:
   - mảng mức hiệu lực;
   - mặt nạ tắt theo chế độ;
   - bộ đệm vòng 60 giá trị RMS và trung vị cho `auto`.

   Thêm khoảng 100 B RAM.
3. Nếu dùng PCEN: `feature_params.h` có `SW_FRONTEND_PCEN 1` và `SW_PCEN_S/GAIN/BIAS/POWER/EPS`; kiểm bằng `golden_pcen.h`
   (`export-golden --frontend pcen`). Chi phí khoảng 49×40 lần `powf` ×2 mỗi hop. Nếu dùng sigmoid: `SW_OUTPUT_SIGMOID 1`, TFLM cần
   kernel LOGISTIC.

## 12. Việc chưa làm

- **Thu dữ liệu thật mới:** ngoài phạm vi.
- **Đánh giá `auto` trên dữ liệu thật:** mức nền trong kịch bản tổng hợp là ngẫu nhiên, xem mục 8.
- **Thử PCEN với hằng số thời gian dài hơn, và sigmoid + trộn với p nhỏ:** hết ngân sách train khoảng 4 giờ (mỗi lần khoảng 50 phút).
- **Không chạy `export_ios_assets.py`,** không sửa Swift, không sửa `ios/SoundWatch/Resources/*`. Model / config của app và của vn1
  giữ nguyên.
