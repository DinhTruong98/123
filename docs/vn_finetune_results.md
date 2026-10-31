# Kết quả train lại SoundWatch với dữ liệu âm thanh Việt Nam

Nhánh `vn-finetune`. Model mới được lưu ở `assets/models/vn/` (`student_int8.tflite`, `feature_stats.json`), dùng cùng cấu hình `config/soundwatch_vn.yaml`. Model đang chạy trong app (`assets/models/student_int8.tflite`) **không bị thay**. Báo cáo chi tiết (json/md) nằm trong `docs/vn_finetune_reports/`.

## Tóm tắt

- **Còi và chuông báo cháy VN được bắt tốt hơn.**
  - Mức cửa sổ 1 s, tập test: recall còi VN 0.83 → **0.93**.
  - Kịch bản chỉ có âm VN: còi 0.87 → **0.91** (test) và 0.89 → 0.89 (val); chuông báo cháy 0.46 → **0.59** (test) và 0.67 → **0.87** (val).
- **Báo nhầm trên nền:** trên kịch bản VN val giảm mạnh (32 → 5 lần/giờ), nhưng trên kịch bản VN test tăng nhẹ (16 → 20). Trên kịch bản tổng hợp, val giữ nguyên (22 → 22), test giảm nhẹ (37 → 35).
- **Dữ liệu gốc:** ở mức cửa sổ, vài lớp tụt **quá 2 điểm**: test baby_cry −4.1, smoke_alarm −4.4; val car_horn −4.6, water_running −6.2. Đây là các lớp ít dữ liệu, nên sai số lớn.
- **Kết luận:** chỉ **đạt một phần** tiêu chí, nên **chưa nên thay model trong app ngay** (xem mục 7).

## 1. Dữ liệu

- **Bộ VN:** 3745 clip theo `data/vn/manifest.csv`.
  - Script `scripts/rebuild_vn_data.py` **không tải được** từ môi trường cloud. YouTube đòi "Sign in to confirm you're not a bot" ở cả 2 lần chạy (lần 2 sau 20 phút, `--workers 1`). Freesound trả HTTP 403 cho mọi request, kể cả trang chủ.
  - Vì vậy đã dùng phương án dự phòng: file tar `vn-sound-data.tar` (1,2 GB) trên Google Drive. File được kiểm tra trước khi giải nén: không có đường dẫn tuyệt đối hoặc `..`; bỏ các file `._*` của macOS và thư mục `tools/`.
  - Đối chiếu với `data/vn/manifest.csv`: đủ 3745/3745 clip, nhãn và split khớp 100%, mọi file đều WAV 16 kHz mono PCM16.
- **Dữ liệu gốc:** ESC-50, UrbanSound8K và FSD50K (chọn lọc) có sẵn trên máy từ lần trước, không tải lại.
- **Build:** `build-dataset --with-fsd50k --custom-manifest ../vn-sound-data/manifest.csv`. Có 3684 clip VN được dùng, 61 clip `label_confidence=low` bị bỏ theo mặc định. `CUSTOM_MAX_WINDOWS` giữ nguyên.

| Lớp | train | val | test | Tỷ lệ VN (train / val / test) |
|---|---|---|---|---|
| smoke_alarm | 916 | 181 | 227 | 63% / 63% / 60% |
| car_horn | 1821 | 148 | 211 | 6% / 14% / 9% |
| **siren** | 7928 | 1189 | 1139 | **31%** / 47% / 46% |
| baby_cry | 604 | 101 | 195 | – |
| door_knock | 108 | 35 | 42 | – |
| dog_bark | 4108 | 568 | 571 | – |
| water_running | 533 | 194 | 188 | – |
| background | 23784 | 3567 | 3417 | 16% / 21% / 20% |

Cửa sổ VN chiếm 31% lớp `siren` ở train, nằm trong mục tiêu 25–35% của kế hoạch.

## 2. Các lệnh đã chạy

```bash
uv run python -m soundwatch build-dataset --with-fsd50k --custom-manifest ../vn-sound-data/manifest.csv
uv run python -m soundwatch train-teacher
uv run python -m soundwatch train-student --distill --threads 4          # công thức mặc định = model trong app (DS-CNN 96→192)
uv run python -m soundwatch quantize --no-copy                            # KHÔNG ghi đè assets/models/student_int8.tflite
uv run python -m soundwatch export && uv run python -m soundwatch export-icons && uv run python -m soundwatch export-golden
uv run python -m soundwatch eval-vn --split {val,test} --model cu assets/models/student_int8.tflite assets/models/feature_stats.json \
    --model moi artifacts/models/student_int8.tflite artifacts/models/feature_stats.json --manifest ../vn-sound-data/manifest.csv
# kịch bản (--no-demo để không ghi đè assets/demo/demo_scenario.wav)
uv run python -m soundwatch make-scenarios --no-demo
uv run python -m soundwatch make-scenarios --split val --out artifacts/data/scenarios_val --no-demo --n-events 30 --n-background 20
uv run python -m soundwatch make-scenarios --split {test,val} --classes danger --event-lengths 0.5,1,1.5 \
    --n-events 20 --n-background 0 --duration 120 --seed 7 --out artifacts/data/scenarios_short_{test,val}
uv run python -m soundwatch make-scenarios --split {test,val} --classes siren,smoke_alarm --source-filter vn_,vn/ \
    --no-demo --n-events 20 --n-background 10 --seed 11 --out artifacts/data/scenarios_vn_{test,val}
uv run python -m soundwatch eval-fast-path --sets test,val,short_test,short_val,vn_test,vn_val --model … --stats …
uv run python -m soundwatch eval-streaming --no-sweep --model … --stats … --out-dir …   # và --scenarios artifacts/data/scenarios_vn_test
uv run python -m soundwatch tune --scenarios artifacts/data/scenarios_val --model artifacts/models/student_int8.tflite …
```

Thay đổi code trên nhánh (commit `d7dc1e0`):
- **Âm nền val:** `build-dataset` giờ ghi thêm `noise/val` (clip nền đầy đủ). Trong lần chạy này, `noise/val` được sinh riêng bằng một script một lần, cùng logic, để không phải build lại.
- `make-scenarios --source-filter` để sinh kịch bản chỉ có âm VN.
- `eval-fast-path --model/--stats` và thêm hai bộ kịch bản `vn_test` / `vn_val`.

## 3. Model

| | Cũ (app) | Mới (`assets/models/vn/`) |
|---|---|---|
| Kiến trúc | DS-CNN 96→192, 4 khối | giống hệt |
| int8 / tham số / phép nhân-cộng | 208 KB / 152 k / 24.5 M | 208 KB / 152 k / 24.5 M |
| Teacher (YAMNet + head), macro-F1 val / test | 0.858 / 0.823 (dữ liệu cũ) | 0.814 / 0.799 |
| Student float, macro-F1 val / test | – | 0.772 / 0.796 |
| int8, macro-F1 val / test (tụt so với float) | – | 0.769 / 0.795 (−0.30 / −0.11 điểm) ✅ |

Teacher trên phần VN: recall siren **0.94** (val) và 0.92 (test); chỉ 2% cửa sổ nền VN bị đoán là siren. Đây là lý do mình distill tiếp từ teacher này.

Tập val/test mới có thêm dữ liệu VN, nên các con số macro-F1 ở trên không so trực tiếp được với số cũ. So sánh công bằng nằm ở mục 4–5, nơi cả hai model chấm trên cùng cửa sổ và cùng kịch bản.

## 4. Mức cửa sổ 1 s (`eval-vn`, cùng cửa sổ cho cả hai model)

| Chỉ số | val cũ | val mới | test cũ | test mới |
|---|---|---|---|---|
| **VN:** macro-F1 | 0.661 | 0.686 | 0.670 | 0.690 |
| **VN:** recall siren | 0.860 | 0.872 | 0.828 | **0.931** |
| **VN:** recall smoke_alarm | 0.561 | **0.842** | 0.664 | 0.686 |
| **VN:** nền → báo động / nền → siren | 0.095 / 0.011 | 0.079 / 0.015 | 0.124 / 0.006 | 0.118 / 0.006 |
| **Gốc:** macro-F1 | 0.805 | 0.787 | 0.784 | 0.793 |
| Gốc: smoke_alarm / car_horn / siren | 0.866 / 0.773 / 0.962 | 0.940 / **0.727** / 0.965 | 0.900 / 0.733 / 0.561 | **0.856** / 0.754 / 0.580 |
| Gốc: baby_cry / door_knock / dog_bark / water_running | 0.851 / 0.771 / 0.754 / 0.727 | 0.842 / 0.829 / 0.762 / **0.665** | 0.815 / 0.857 / 0.779 / 0.713 | **0.774** / 0.881 / 0.795 / 0.755 |

Số in đậm ở hàng "Gốc" là các lớp tụt quá 2 điểm.

**Recall còi VN theo nhóm (test):**

| Nhóm | Cũ | Mới |
|---|---|---|
| ambulance (144 cửa sổ) | 0.972 | 0.938 |
| fire (234 cửa sổ) | 0.688 | **0.902** |
| police (108 cửa sổ) | 0.889 | 0.972 |
| other (36 cửa sổ) | 0.972 | 0.972 |
| passing, xe đi ngang qua (24 cửa sổ) | 0.750 | **0.917** |
| far / mid / near | 0.904 / 0.792 / 0.798 | 0.942 / 0.970 / 0.889 |

Trên val, mức tăng nhỏ hơn: fire 0.867 → 0.908, police 0.812 → 0.840, passing 0.741 → 0.741, ambulance 0.882 → 0.833. Bảng đầy đủ: `vn_eval_{val,test}.md`.

## 5. Mức sự kiện (cùng kịch bản cho cả hai model)

Cấu hình hậu xử lý: model cũ dùng `config/soundwatch.yaml`, model mới dùng `config/soundwatch_vn.yaml`. Hai file chỉ khác ngưỡng siren (0.99 so với 0.95, xem mục 6). Cả hai đều có `danger_fast_threshold` 255/256.

| Bộ kịch bản | Recall tổng cũ → mới | DANGER TB | smoke / car_horn / siren (cũ → mới) | Báo nhầm/giờ trên nền | Trễ median (s) |
|---|---|---|---|---|---|
| test (tổng hợp, 86 sự kiện, 1 giờ nền) | 0.42 → 0.44 | 0.56 → 0.53 | 0.42 → 0.42 / 0.58 → 0.58 / 0.67 → 0.58 | 37 → **35** | 1.48 → 1.41 |
| val (130 sự kiện, 2 giờ nền) | 0.48 → 0.52 | 0.56 → 0.61 | 0.68 → 0.84 / 0.26 → 0.32 / 0.74 → 0.68 | 22 → **22** | 1.36 → 1.23 |
| **VN test** (91 sự kiện còi + báo cháy VN, nền VN 1 giờ) | 0.66 → **0.75** | 0.66 → 0.75 | 0.46 → **0.59** / – / 0.87 → **0.91** | 16 → **20** | 1.43 → 1.23 |
| **VN val** (90 sự kiện) | 0.78 → **0.88** | 0.78 → 0.88 | 0.67 → **0.87** / – / 0.89 → 0.89 | 32 → **5** | 1.49 → 1.57 |
| DANGER ngắn, test (0.5/1/1.5 s) | 0.34 → 0.33 | – | 0.43 → 0.50 / 0.34 → 0.24 / 0.24 → 0.24 | – | 0.79 → 0.81 |
| DANGER ngắn, val | 0.34 → 0.33 | – | 0.30 → 0.48 / 0.30 → 0.18 / 0.42 → 0.33 | – | 0.92 → 0.91 |

Cách đọc bảng:
- Siren trong bộ test tổng hợp chỉ có 12 sự kiện, nên chênh 1 sự kiện là ±0.08.
- Báo nhầm trên nền VN test tăng từ 16 lên 20 lần/giờ, chủ yếu do `dog_bark` (10 → 13). Báo nhầm siren chỉ 1 → 2, smoke_alarm 0 → 2.
- Trên test tổng hợp, báo nhầm siren giảm 17 → 13.
- Báo nhầm của các lớp WARN/INFO (dog_bark, baby_cry) không phụ thuộc vào ngưỡng siren đã chỉnh.

## 6. Chỉnh ngưỡng (chỉ trên val)

- **Vì sao ngưỡng 0.99 không còn hợp:** model mới **ít bão hoà hơn** model cũ. Trên còi VN ở test, tỷ lệ cửa sổ có xác suất ≥ 0.9 là 0.72 (cũ) và 0.81 (mới), nhưng tỷ lệ bão hoà ≥ 0.996 là 0.50 (cũ) và 0.43 (mới). Vì vậy ngưỡng siren 0.99, vốn chọn cho model cũ, làm mất sự kiện.
- **Kết quả `tune` đầy đủ** (`tuning_moi.md`): không có cấu hình nào vừa có trễ median ≤ 1.5 s vừa có ≤ 10 lần báo nhầm/giờ. Kết luận giống model cũ ở WP8, và báo nhầm chủ yếu đến từ dog_bark và baby_cry. Vì vậy mình giữ nguyên smoothing, k-of-n và ngưỡng các lớp khác, chỉ quét ngưỡng siren (`siren_sweep_val.json`):

| Ngưỡng siren (model mới) | val: siren / báo nhầm/giờ | VN val: siren / báo nhầm/giờ |
|---|---|---|
| 0.90 | 0.74 / 24.5 | 0.89 / 7 |
| **0.95** | **0.68 / 22.0** | **0.89 / 5** |
| 0.97 | 0.58 / 21.0 | 0.82 / 5 |
| 0.99 (giá trị cũ) | 0.58 / 20.5 | 0.78 / 5 |

  Chọn **0.95**: đây là mức có recall siren cao nhất mà báo nhầm trên val không vượt model cũ (22/giờ). Mức 0.90 cho recall cao hơn nhưng lên 24.5 lần/giờ.
- **`danger_fast_threshold`:** vẫn giữ 255/256. Trên val, mức 254/256 cho recall gần bằng nhưng báo nhầm trên VN val tăng từ 5 lên 7 lần/giờ.

## 7. Các lần thử

| Lần | Thay đổi | Kết quả | Quyết định |
|---|---|---|---|
| 0 | Tải lại dữ liệu bằng `rebuild_vn_data.py` (2 lần) | 0/3745 clip: YouTube đòi xác minh bot, Freesound trả 403 | Dùng file tar trên Drive |
| 1 | Công thức mặc định + distill | Như mục 3–6 | **Giữ** (`assets/models/vn/`) |
| 2 | `--low-snr-copies 1` (trộn sự kiện, gồm còi VN, vào nền gồm nền VN ở SNR 0–10 dB) | Macro-F1 float val 0.754 / test 0.770, đều kém lần 1 (0.772 / 0.796); recall car_horn val chỉ 0.51 | Bỏ, không quantize |
| Đo | Kịch bản val dựng từ cửa sổ 1 s (cách cũ) | Báo nhầm smoke_alarm giả trên nền VN (24 lần/giờ). Nguyên nhân: cửa sổ nền bị giới hạn số lượng nên các đoạn 0.5 s bị ghép không liền mạch | Sửa: dựng nền val từ clip đầy đủ (`noise/val`). Mọi số val ở trên đã dùng cách mới |

Chưa thử tăng hay giảm `CUSTOM_MAX_WINDOWS`, vì tỷ lệ VN của siren đã nằm trong mục tiêu và mỗi lần train mất khoảng 25 phút.

Không phát hiện clip VN nào bị gán nhãn sai chắc chắn. Các báo nhầm smoke_alarm trên nền chợ (`yt-CQgoLzpYKDY`, `yt-EHvTNB1gr84`) chỉ xuất hiện khi âm nền bị ghép mảnh như mô tả ở trên. Khi chấm trực tiếp từng cửa sổ, chúng không bị đoán là báo động.

## 8. Đánh giá theo tiêu chí và đề xuất

| Tiêu chí | Kết quả |
|---|---|
| Recall còi VN tăng rõ | ✅ ở test (cửa sổ 0.83 → 0.93; sự kiện 0.87 → 0.91; xe cứu hoả 0.69 → 0.90; xe đi ngang qua 0.75 → 0.92). ⚠️ ở val chỉ nhích nhẹ (0.86 → 0.87; sự kiện 0.89 → 0.89) |
| Chuông báo cháy VN | ✅ tăng ở mọi bộ (sự kiện VN test 0.46 → 0.59, VN val 0.67 → 0.87) |
| Recall lớp khác trên dữ liệu gốc không tụt quá 2 điểm | ❌ tụt quá 2 điểm: baby_cry và smoke_alarm (test), car_horn và water_running (val) ở mức cửa sổ. Các lớp này ít dữ liệu (42–195 cửa sổ), nên một phần là nhiễu |
| Báo nhầm/giờ trên nền không tăng | ⚠️ val tổng hợp 22 → 22, VN val 32 → 5, test tổng hợp 37 → 35, nhưng **VN test 16 → 20** |

**Đề xuất:**
- **Chưa thay** model trong app. Model mới tốt hơn rõ với còi xe cứu hoả, xe đi ngang qua và chuông báo cháy VN, nhưng chưa đạt tiêu chí "không tụt lớp khác" và "không tăng báo nhầm". Vì vậy mình **không** chạy `export_ios_assets.py`.
- Nên thử model mới song song trên iPhone (đổi `soundwatch_config.json` sang model và ngưỡng của `config/soundwatch_vn.yaml`) để đo với còi thật ngoài đường trước khi quyết định.
- **Hướng cải thiện tiếp:**
  1. Thêm âm nền VN có chó sủa hoặc tiếng người, để giảm báo nhầm `dog_bark` trên nền VN.
  2. Tăng dữ liệu cho baby_cry, door_knock và water_running để các lớp này không tụt khi thêm dữ liệu VN.
  3. Thử `CUSTOM_MAX_WINDOWS` nhỏ hơn cho lớp `smoke_alarm`. VN đang chiếm 63% lớp này, nhiều khả năng đây là lý do smoke_alarm trên dữ liệu gốc ở test tụt 4.4 điểm.
