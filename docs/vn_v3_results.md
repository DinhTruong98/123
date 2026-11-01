# SoundWatch VN v3: còi hú / chuông báo động theo âm thanh + hiển thị theo ngữ cảnh

Nhánh `vn-v3`, xây từ `vn-v2`. Model mới ở `assets/models/vn_v3/` (`student_int8.tflite`, `feature_stats.json`), dùng với
`config/soundwatch_vn_v3.yaml`. Model trong app (**cu**, `assets/models/student_int8.tflite`), **vn1** (`assets/models/vn/`) và
**vn2** (`assets/models/vn_v2/`) **không bị thay**. Báo cáo chi tiết (json/md) ở `docs/vn_v3_reports/`.

## Tóm tắt

- **Định nghĩa lại hai lớp DANGER theo âm thanh** (key và hợp đồng 8 lớp giữ nguyên): `siren` = **"Còi hú"** (mọi còi hú / quét /
  hai tông, kể cả còi báo cháy kiểu hú), `smoke_alarm` = **"Chuông / bíp báo động"**. Đã xem spectrogram cả 158 clip
  `smoke_alarm` VN: **44 clip → `siren`**, **11 clip bỏ** (lẫn cả hai âm), 103 giữ nguyên. Ghi ở `data/vn/relabel_v3.csv` (manifest
  không sửa), dùng bằng `build-dataset --custom-relabel`. Chiều ngược lại (siren VN thật ra là chuông/bíp) và FSD50K: không clip nào.
- **Model v3** = công thức vn2 (DS-CNN 96→192 softmax log-mel, distill, teacher train lại, hiệu chỉnh nhiệt độ T = 1.83 gộp vào
  Dense cuối), cùng kích thước / op / MACs / arena. Ngưỡng chọn **chỉ trên val** bằng đúng quy trình vn2.
- **Nhầm còi hú ↔ chuông/bíp (chỉ số chính, nhãn mới):**
  - Mức cửa sổ, VN val: siren → smoke **4.7% (vn2) / 9.3% (vn1) → 0.7%**; trên các clip còi báo cháy kiểu hú (đã đổi sang siren):
    **57% (vn2) / 63% (vn1) → 0%**. smoke → siren ≈ 0–1% ở mọi model (sau khi đổi nhãn, `smoke_alarm` chỉ còn chuông/bíp).
  - Mức hop trong sự kiện (kịch bản): tỉ lệ hop tự tin mà lớp kia thắng, sự kiện siren val **5.3% (vn2) → 0.5%**, VN val 1.0% → 0%.
  - Mức sự kiện: sự kiện siren bị báo thành chuông/bíp: vn2 1/20 (val), 1/47 (VN val) → **v3 0** ở cả sáu bộ. Sự kiện "lật" (cả hai
    lớp cùng báo trong một sự kiện): **0** ở v3 (cu 1, vn1 1 trên val). Ngược lại v3 có 1/46 sự kiện chuông VN test bị báo "Còi hú".
  - "Có báo DANGER" (siren ∪ smoke ∪ car_horn) không mất: sự kiện VN test 0.80 (smoke) / 0.89 (siren), VN val 0.96 / 0.91 (vn2: 0.72 /
    0.91, 0.92 / 0.89).
- **Lớp khác / báo nhầm:** điểm val (recall có trọng số, val ∪ VN val ∪ DANGER-ngắn-val) cu 0.591 / vn2 0.656 → **v3 0.692**.
  Báo nhầm nền val 22 → 20, VN val 16 → **4**/giờ. **Nhưng** báo nhầm "Còi hú" trên nền tăng (val 7 → 17 lần/2 giờ, VN test 1 → 11
  lần/giờ; VN test tổng 19 → 33/giờ, ở chế độ `street` 16/giờ) và `smoke_alarm` gốc (đồng hồ báo thức + FSD50K, 67–90 cửa sổ)
  tụt rõ ở mức cửa sổ (test 0.844 → 0.700, val 0.970 → 0.746, đoán thành nền). Một lần train khác chỉ đổi seed cho 0.867 / 0.821, tức
  dao động seed lớn ngang mức tụt.
- **Hiển thị theo ngữ cảnh** (tuỳ chọn, tương thích ngược): `modes.<mode>.label/icon`; trong nhà "Còi hú báo động / báo cháy" (icon
  lửa), ngoài đường "Xe ưu tiên (cấp cứu / cảnh sát / cứu hoả)". Có trong simulator, console `run`, log, `labels.h`, JSON iOS.
- **`auto` dựa vào âm nền:** EMA của p(car_horn) − p(baby_cry) − p(door_knock) − p(water_running) (1 số float), kết hợp trung vị
  RMS, trễ và thời gian ở tối thiểu 60 s. Trên phiên thử test (xen kẽ 3 phút trong nhà / 3 phút ngoài đường): khi mức âm **không**
  mang thông tin, RMS cũ đúng **50–54%** thời gian (= đoán bừa), bằng chứng âm thanh **77–82%** (sau 60 s đầu mỗi đoạn: 91–96%),
  đổi chế độ sau trung vị 30–34 s. Khi mức âm giả định tách được trong nhà / ngoài đường: RMS một mình 87–89%, kết hợp 83–84%
  (sau 60 s: 97–99%), 0–0.6 lần đổi sai/giờ; tức âm thanh giúp khi mức âm không đáng tin, không thay được một ngưỡng dBFS tốt.
  **Ngưỡng dBFS vẫn phải đo lại với micro thật** của đồng hồ / iPhone.
- **Đề xuất:** dùng v3 + `soundwatch_vn_v3.yaml` làm ứng viên thử trên iPhone **cạnh vn2** (nhánh so sánh hai model), ưu tiên
  kiểm: clip còi xe ưu tiên / còi báo cháy kiểu hú (không còn nhảy nhãn) và báo nhầm "Còi hú" ngoài đường. Chưa thay model mặc định
  của app (mục 10).

## 1. Vì sao

Khi thử trên iPhone với clip TikTok xe cấp cứu / cảnh sát / chuông báo cháy, app nhảy qua lại giữa "Còi cấp cứu" (`siren`) và
"Báo cháy" (`smoke_alarm`). Nguyên nhân nằm ở **định nghĩa nhãn**: `smoke_alarm` được gán theo *nghĩa* (thiết bị báo cháy), nên
gồm cả những âm *nghe y hệt* còi xe cấp cứu:
- còi báo cháy kiểu mô-tơ ("Còi báo động bằng motor … Còi báo cháy"), còi hú điện tử của tủ báo cháy / đèn chớp
  (tiếng "yelp" quét lên xuống 2–4 lần/giây);
- video diễn tập PCCC, trong đó tiếng xe cứu hoả hú át hết chuông báo cháy.

Về âm học, còi hú báo cháy và còi xe cấp cứu là **cùng một loại âm** (một tông quét lên xuống). Model buộc phải học "quét lên
xuống ở clip này là báo cháy, ở clip kia là cấp cứu", nên với âm hú ngoài đời nó đoán nửa nọ nửa kia, và kết quả hiển thị bị nhảy.

**Quyết định (đã được duyệt): phương án B + hiển thị theo ngữ cảnh.**
1. Hai lớp DANGER được định nghĩa **theo âm thanh**, giữ nguyên key và hợp đồng 8 lớp (app iOS, firmware, test vẫn chạy):
   - `siren` = **"Còi hú"**: mọi âm còi hú / quét / hai tông (cấp cứu, cảnh sát, cứu hoả, còi báo cháy mô-tơ hay điện tử, còi
     phòng không).
   - `smoke_alarm` = **"Chuông / bíp báo động"**: chuông điện, đầu báo khói bíp, còi-đèn kêu bíp (tông đứng, ngắt quãng), chuông
     đồng hồ báo thức. ESC-50 `clock_alarm` (đồng hồ báo thức, dùng làm proxy) **vẫn ở lại lớp này**: theo âm thì nó là tiếng bíp
     báo động. Cái giá: model sẽ báo "Chuông / bíp báo động" cho đồng hồ báo thức; với người khiếm thính đây là thông tin hữu ích,
     nhưng không phải "cháy".
2. **Nghĩa** của âm (báo cháy hay xe ưu tiên) do **ngữ cảnh** quyết định: trong nhà, còi hú nhiều khả năng là còi báo cháy; ngoài
   đường, nhiều khả năng là xe ưu tiên. Mỗi chế độ (`indoor` / `street`) có nhãn và icon riêng, và chế độ `auto` nay dựa vào
   **âm nền** (model tự nghe thấy còi xe hay âm sinh hoạt trong nhà), không chỉ dựa vào độ ồn.

## 2. Gán lại nhãn (`data/vn/relabel_v3.csv`)

`data/vn/manifest.csv` là nguồn đã khoá, **không sửa**. Thay vào đó có file ghi đè `data/vn/relabel_v3.csv`
(`file,old_class,new_class,reason,siren_score,alarm_score,title`; `new_class = drop` là bỏ clip) và tuỳ chọn mới
`build-dataset --custom-relabel <csv>` (`data/build_dataset.py`: `load_relabel`, `apply_relabel`). Tuỳ chọn này kiểm `old_class`
khớp nhãn hiện tại, và clip VN đổi lớp nhận giới hạn cửa sổ của lớp mới (vd siren 6 cửa sổ/clip thay vì smoke 2 ở train). Cột
`file` cũng nhận id của bộ khác (`fsd50k/…`, `esc50/…`), nhưng lần này không cần.

**Cách quyết định:** vẽ spectrogram (0–5 kHz) của **toàn bộ 158 clip** `smoke_alarm` VN (`scripts/vn_v3/spectro.py`, 11 trang ảnh
trong `artifacts/vn_v3/spec/`, xem bằng mắt từng clip), đối chiếu tiêu đề video và điểm YAMNet trong manifest:
- tông đứng ngắt quãng (bíp 3 kHz của đầu báo khói), tông đứng liên tục, chuông (nhiều hài, đứng) → **giữ `smoke_alarm`**;
- tông quét chậm lên xuống (mô-tơ, xe cứu hoả) hoặc quét nhanh lặp lại (yelp điện tử, còi hú đèn chớp) → **`siren`**;
- clip có **rõ cả hai** (vd yelp của xe cứu hoả + khối bíp 3.5 kHz của còi-đèn trong video diễn tập) → **bỏ** (không ép nhãn).

Ví dụ: `docs/vn_v3_reports/spec_smoke_02.jpg` (bíp, chuông, yelp, mô-tơ), `spec_smoke_06.jpg` (diễn tập: hú + clip lẫn tông
đứng bị bỏ), `spec_smoke_09.jpg` (diễn tập Dĩ An: yelp + khối bíp → bỏ).

**Kết quả** (55 dòng trong csv; clip `label_confidence = low` vốn đã bị loại khi build, ghi vào csv cho đủ):

| | train | val | test | tổng | (trong đó low) |
|---|---|---|---|---|---|
| smoke_alarm → **siren** | 33 | 6 | 5 | **44** | 4 |
| smoke_alarm → **bỏ** (có cả hai âm) | 9 | 0 | 2 | **11** | 1 |
| giữ smoke_alarm | | | | 103 | |

- **Cả 26 clip** có `siren_score > alarm_score` đều được xử lý: 20 → siren, 6 → bỏ (diễn tập Dĩ An ×5 có khối bíp còi-đèn,
  "dien tap pccc tcvl 3" ×1 có đoạn bíp ở cuối).
- **18 clip nữa** được phát hiện **nhờ spectrogram** dù YAMNet nghiêng về "alarm": còi báo động 24V H207X (test ×3), còi không dây
  GS-WS26 (test ×2), Kingfood và "Chuông báo cháy tự động" (val ×5, chuông điện tử kêu quét răng cưa), còi hú đèn chớp 103 (×4),
  còi KM-628, còi remote, diễn tập Chutex. YAMNet gọi các âm này là "alarm" vì nó cũng học theo nghĩa.
- 5 clip khác bị bỏ vì lẫn hai âm: "Thực hiện phương án chữa cháy" ×2 (tông 4 kHz + đoạn hú), "PCCC An Tâm Phát" ×2 (bíp 2.8 kHz +
  còi-đèn quét, **test**), "Hướng dẫn test nhanh" ×1 (low).
- **Chiều ngược lại:** nghe/xem 24 clip `siren` VN có điểm "alarm" gần điểm "siren" nhất (`artifacts/vn_v3/spec/siren_lowmargin_*`):
  tất cả đều là còi hú / hai tông của xe ưu tiên (đèn còi CSGT, ampli còi hú Senken, hi-lo xe cấp cứu). **Không** đổi clip nào.
- **FSD50K `smoke_alarm`** (18 clip đã giữ, `spec_fsd_smoke_00.jpg`): toàn bíp / chuông, không clip nào hú. ESC-50 `clock_alarm`:
  bíp đồng hồ, giữ nguyên (mục 1).

Thay đổi số cửa sổ (`artifacts/data/windows/summary.json`):

| | VN smoke train / val / test | VN siren train / val / test |
|---|---|---|
| vn2 | 192 / 114 / 137 | – / 564 / 522 |
| **v3** | **116 / 84 / 95** | **2676 / 594 / 552** |

Cửa sổ nền val/test thay đổi nhẹ vì giới hạn nền = 3 × lớp lớn nhất (siren) đổi theo (val 5983 → 6073 cửa sổ, test 5990 → 6068).
Dữ liệu gốc (ESC-50/US8K/FSD50K) giữ đúng nhãn cũ.

## 3. Hiển thị theo ngữ cảnh (lớp sản phẩm, tuỳ chọn, tương thích ngược)

### 3.1 Nhãn / icon theo chế độ (quy tắc 9b)

Mỗi chế độ trong `modes:` có thể ghi đè **cách hiển thị** một lớp (khoá `label`, `icon`); model và luật báo động không đổi:

```yaml
modes:
  default: indoor
  indoor:
    label: {siren: "Còi hú báo động / báo cháy", smoke_alarm: "Chuông báo cháy / báo động"}
    icon: {siren: fire.png}
  street:
    disable: [dog_bark, baby_cry]
    level: {car_horn: 0}
    label: {siren: "Xe ưu tiên (cấp cứu / cảnh sát / cứu hoả)", smoke_alarm: "Chuông báo động"}
    icon: {siren: siren.png}
```

- `config.ModeSpec` có thêm `label` / `icon` (kiểm: lớp phải tồn tại, nhãn không rỗng, icon có trong `assets/icons` khi
  `check_icons`). Nhãn gốc của lớp trong v3 là nhãn **theo âm**: `siren` "Còi hú", `smoke_alarm` "Chuông / bíp báo động".
- `PostProcessor`: khi đổi chế độ, tính nhãn/icon hiệu lực; mỗi `AlertEvent` mang `label` / `icon` của chế độ **lúc báo** (None
  khi không ghi đè). `PostProcessor.display(c)` trả (nhãn, icon) hiệu lực. Hàm `types.event_label/event_icon` dùng cho UI.
- Mọi đầu ra đều mang nhãn hiệu lực: simulator pygame (màn cảnh báo, viên "Chưa xác nhận", sự kiện gần nhất, lịch sử; nhãn dài
  được xuống tối đa 3 dòng), dòng `run` trên console, log sự kiện (`label_vi` = nhãn hiệu lực, thêm `icon` khi có ghi đè),
  `labels.h` (`kModeLabelUtf8/Ascii`, `kModeIconName[mode][class]`), JSON iOS (`modes.modes.<mode>.label/icon/sf_symbol`).
- Simulator: dòng trạng thái trên mặt đồng hồ hiện chế độ ("Ngoài đường (tự động)"), phím **M** chuyển vòng
  indoor → street → auto (gửi `ModeCommand`). `StatusUpdate` có thêm `mode` / `mode_selector`.
- **Tương thích ngược:** không có `label`/`icon` thì sự kiện giống hệt (test `test_no_display_overrides_identical_events`). Với
  `config/soundwatch.yaml` + model cu: 12 sự kiện demo và xác suất từng hop trùng tuyệt đối với
  `ios/SoundWatch/Resources/demo_expected_events.json`, `soundwatch_config.json` sinh lại trùng file trong `Resources`
  (`scripts/vn_v3/check_compat.py`, không ghi file nào). JSON và `labels.h` của `soundwatch_vn_v2.yaml` không có khoá mới.

Ảnh simulator: `docs/vn_v3_reports/faces_street.png` (chế độ ngoài đường: cảnh báo "Xe ưu tiên (cấp cứu / cảnh sát / cứu hoả)"
hai dòng với icon còi, lịch sử, mặt chờ "Ngoài đường (tự động)" với viên "Chưa xác nhận"). Lệnh thử:
`run --config config/soundwatch_vn_v3.yaml --source assets/demo/demo_scenario.wav --headless` in "Chuông báo cháy / báo động",
"Còi hú báo động / báo cháy"; thêm `--mode street` in "Chuông báo động", "Xe ưu tiên (cấp cứu / cảnh sát / cứu hoả)".

### 3.2 `auto` dựa vào âm nền

vn2 chỉ dùng trung vị RMS 30 s so với −35 dBFS. Cách này **không phân biệt được** trong nhà ồn (TV, karaoke, quán) với ngoài đường,
và phụ thuộc hoàn toàn vào độ nhạy micro. v3 thêm **bằng chứng âm thanh từ chính model** (`config.AutoModeConfig`):

```
x_t          = Σ_k w_k · p_k(t)          w = {car_horn: +1, baby_cry: −1, door_knock: −1, water_running: −1}
street_score = EMA(x_t, hằng số thời gian evidence_window_s)        # 1 số float, MCU rẻ
street  khi street_score ≥ street_score_above
indoor  khi street_score <  street_score_above − score_hysteresis
có cả RMS: street cần thêm "không yên tĩnh" (trung vị RMS ≥ ngưỡng − trễ/2), indoor cần "không ồn"
không quyết định trước min_s, không tự đổi lại trước min_dwell_s (thời gian tối thiểu ở một chế độ)
```

Ý tưởng: ngoài đường VN, model liên tục nghe thấy "một chút" còi xe (p(car_horn) trung bình 0.03 so với 0.01 trong nhà) ngay
cả khi không có tiếng bấm còi rõ; trong nhà, các âm sinh hoạt kéo p(baby_cry / door_knock / water_running) lên. Dùng **xác suất
mềm** quan trọng: bản đầu tiên đếm tỉ lệ hop có p(car_horn) ≥ x (AUC chỉ 0.54–0.67 trên val), bản xác suất mềm đạt AUC
0.98–0.99 trên val (cửa sổ 30 s, đo trên vn2). Không dùng `siren` làm bằng chứng đường phố: còi báo cháy trong nhà nay cũng là
"siren".

Mọi tham số mới là tuỳ chọn; thiếu `street_evidence` thì `auto` chạy đúng như vn2 (`test_auto_rms_only_unchanged`).

## 4. Các lệnh đã chạy

```bash
git checkout -b vn-v3 origin/vn-v2
P="uv run python -m soundwatch"
# 1. xem lại nhãn: spectrogram của mọi clip smoke_alarm VN, 24 clip siren VN gần "alarm" nhất, 18 clip FSD50K smoke
uv run python scripts/vn_v3/spectro.py artifacts/vn_v3/spec/smoke 15 <file::caption> ...
uv run python scripts/vn_v3/make_relabel.py data/vn/manifest.csv data/vn/relabel_v3.csv     # quyết định ghi trong script
# 2. dữ liệu (≈ 30 phút, đĩa lạnh)
$P build-dataset --with-fsd50k --custom-manifest /home/user/vn-sound-data/manifest.csv --custom-relabel data/vn/relabel_v3.csv
# kịch bản: đúng các lệnh ở docs/vn_finetune_results.md mục 2, luôn --no-demo
$P make-scenarios --no-demo
$P make-scenarios --split val --out artifacts/data/scenarios_val --no-demo --n-events 30 --n-background 20
$P make-scenarios --split {test,val} --classes danger --event-lengths 0.5,1,1.5 --n-events 20 --n-background 0 \
    --duration 120 --seed 7 --out artifacts/data/scenarios_short_{test,val} --no-demo
$P make-scenarios --split {test,val} --classes siren,smoke_alarm --source-filter vn_,vn/ --no-demo --n-events 20 \
    --n-background 10 --seed 11 --out artifacts/data/scenarios_vn_{test,val}
# 3. train (2 run song song, mỗi run --threads 2, ≈ 90 phút vì máy bận)
$P train-teacher
$P train-student --distill --threads 2 --tag v3_sm_lm                                     # A = v3
$P train-student --distill --threads 2 --seed 7 --cache-dir artifacts/vn_v3/cache_seed7 --tag v3_sm_lm_s7   # B: chỉ đổi seed
$P quantize --tag <tag> --no-copy --calibrate --out-dir artifacts/models/<tag> [--cache-dir …]
# 4. chọn hậu xử lý trên val (công cụ vn2; SW_RUNS = thư mục cache mới)
export SW_RUNS=artifacts/reports/v3/runs
uv run python scripts/vn_v2/runs_cache.py <name> <config> <model> <stats>      # cu, vn1, vn2, v3_sm_lm, v3_sm_lm_s7
uv run python scripts/vn_v2/select_pp.py <tag> config/soundwatch_vn_v3.yaml artifacts/reports/v3/selectF_<tag>.json 22 32 0.05
uv run python scripts/vn_v2/make_cfg.py artifacts/reports/v3/selectF_v3_sm_lm.json config/soundwatch_vn_v3.yaml \
    config/soundwatch_vn_v3.yaml assets/models/vn_v3            # + car_horn/siren 255/256 -> 0.9 (giống hệt trên val)
# 5. đánh giá
$P eval-vn --split {val,test} --manifest /home/user/vn-sound-data/manifest.csv --relabel data/vn/relabel_v3.csv \
    --model cu … --model vn1 … --model vn2 … --model v3 … --model v3s7 …
uv run python scripts/vn_v2/compare.py out.json cu=config/soundwatch.yaml vn1=config/soundwatch_vn.yaml \
    vn2=config/soundwatch_vn_v2.yaml vn2=config/soundwatch_vn_v2.yaml@street v3_sm_lm=config/soundwatch_vn_v3.yaml \
    v3_sm_lm=config/soundwatch_vn_v3.yaml@street
uv run python scripts/vn_v3/confusion.py out.json cu=config/soundwatch.yaml:<model>:<stats> …   # chỉ số chính
$P eval-fast-path --config config/soundwatch_vn_v3.yaml --values 0.85,0.9,0.95,0.97,0.99609375
uv run python scripts/vn_v3/auto_mode_eval.py out.json v3=<model>:<stats>                      # chế độ auto
uv run python scripts/vn_v3/check_compat.py                                                     # tương thích ngược
$P export --config config/soundwatch_vn_v3.yaml --out artifacts/firmware_vn_v3                  # không commit
```

## 5. Thí nghiệm và chọn model

| Lần | Model | int8 macro-F1 val / test (cửa sổ) | Tụt float→int8 (điểm, val / test) | T | Hậu xử lý chọn trên val | val: recall / báo nhầm/giờ | VN val: recall / báo nhầm/giờ | DANGER ngắn val | **Điểm val** |
|---|---|---|---|---|---|---|---|---|---|
| | cu (config gốc) | – | – | – | 255/256, 2/2 | 0.478 / 22 | 0.895 / 32 | 0.383 | 0.591 |
| | vn2 (config vn2) | – | – | 2.11 | 0.95, 2/3 | 0.594 / 22 | 0.895 / 16 | 0.455 | 0.656 |
| **A** | **v3 = công thức vn2, nhãn mới** | 0.770 / 0.794 | −0.10 / −0.03 | 1.83 | fast 0.9, 2/3 | **0.638 / 20** | **0.937 / 4** | **0.487** | **0.692** |
| B | như A, seed 7 | 0.758 / 0.794 | +0.28 / +0.09 | 1.85 | fast 0.97, 2/3 | 0.565 / 18.5 | 0.958 / 8 | 0.412 | 0.656 |

(cu, vn2 dùng config sẵn có, không chọn lại ngưỡng trên dữ liệu mới. Macro-F1 cửa sổ int8 của A/B trên toàn bộ val/test, nhãn mới;
"tụt" theo quy ước của `quantize`. Teacher train lại trên nhãn mới: macro-F1 val / test 0.819 / 0.807, vn2: 0.817 / 0.810.)

Chọn **A** theo điểm val. B cho thấy **dao động giữa hai seed rất lớn** ở các lớp ít dữ liệu: smoke_alarm gốc
test 0.700 (A) so với 0.867 (B), door_knock val 0.743 / 0.657, car_horn val 0.734 / 0.664. Mọi so sánh lớp nhỏ (< 100 cửa sổ) dưới
đây nên đọc với sai số cỡ ±10 điểm.

**Cấu hình chọn cho v3** (`config/soundwatch_vn_v3.yaml`): smooth 1, k-of-n **2/3**, `danger_fast_threshold` **0.9**; ngưỡng smoke
**0.45**, car_horn 0.9, siren 0.9 (quy trình chọn 255/256 cho hai lớp này, tức chỉ báo qua luật hai tầng; 0.9 cho kết quả **giống
hệt** trên cả 6 bộ và vẫn báo được khi tắt luật hai tầng), baby 0.95, door 0.30, dog 0.95, water 0.65.

Quét `danger_fast_threshold` (các ngưỡng khác giữ nguyên, `eval-fast-path`; val để chọn, test chỉ báo cáo):

| fast | val: recall / báo nhầm/giờ | VN val | DANGER ngắn val | (test) | (VN test) |
|---|---|---|---|---|---|
| tắt | 0.594 / 15.5 | 0.916 / 1 | 0.330 | 0.488 / 26 | 0.815 / 26 |
| 0.85 | 0.645 / 25.0 (vượt 22) | 0.947 / 7 | 0.526 | 0.512 / 38 | 0.848 / 38 |
| **0.90** | **0.638 / 20.0** | **0.937 / 4** | **0.485** | 0.500 / 34 | 0.826 / 33 |
| 0.95 | 0.623 / 16.0 | 0.916 / 2 | 0.423 | 0.500 / 29 | 0.826 / 30 |
| 255/256 | 0.594 / 15.5 | 0.916 / 1 | 0.340 | 0.488 / 26 | – |

## 6. Chỉ số chính: nhầm còi hú ↔ chuông/bíp (nhãn mới)

**Lưu ý:** cu / vn1 / vn2 được train với **nghĩa cũ** (còi báo cháy kiểu hú = `smoke_alarm`). Ở đây chúng được chấm theo **nhãn
mới**, nên số của chúng trên các clip đã đổi nhãn đo đúng hiện tượng "còi hú bị gọi là báo cháy" mà người dùng thấy, không phải
"lỗi" của model cũ theo định nghĩa của chính nó. Với nhãn cũ, tài liệu trước đo được smoke → siren 15% (cu) / 11% (vn2) ở VN test;
phần lớn số đó chính là các clip nay đã sang `siren`.

### 6.1 Mức cửa sổ (`scripts/vn_v3/confusion.py`, `docs/vn_v3_reports/confusion_siren_smoke.json`)

| | VN val: smoke→siren / siren→smoke | VN test | VN, chỉ các clip đã đổi sang siren (val 30 / test 30 cửa sổ): đoán thành smoke | Gốc val: smoke→siren / siren→smoke | Gốc test |
|---|---|---|---|---|---|
| cu | 0.000 / 0.002 | 0.000 / 0.000 | 0.00 / 0.00 | 0.000 / 0.000 | 0.011 / 0.015 |
| vn1 | 0.000 / **0.093** | 0.000 / 0.027 | **0.63** / 0.03 | 0.000 / 0.000 | 0.000 / 0.021 |
| vn2 | 0.012 / **0.047** | 0.000 / 0.002 | **0.57** / 0.00 | 0.000 / 0.000 | 0.000 / **0.041** |
| **v3** | 0.012 / **0.007** | 0.011 / **0.000** | **0.00 / 0.00** | 0.000 / 0.000 | 0.000 / **0.005** |

"DANGER có báo" (cửa sổ siren / smoke / car_horn đoán thành một trong ba lớp đó):

| | VN val | VN test | Gốc val | Gốc test | clip đã đổi nhãn val / test |
|---|---|---|---|---|---|
| cu | 0.837 | 0.820 | 0.930 | 0.654 | 0.87 / 0.67 |
| vn1 | 0.913 | 0.922 | 0.930 | 0.663 | 0.97 / 0.67 |
| vn2 | 0.926 | 0.942 | 0.945 | **0.690** | 0.97 / 0.50 |
| **v3** | 0.911 | 0.930 | 0.928 | 0.643 | 0.97 / 0.63 |

Nhận xét: v3 gần như xoá nhầm siren → smoke (VN val 4.7% → 0.7%, gốc test 4.1% → 0.5%), và các còi báo cháy kiểu hú (Kingfood,
"chuông báo cháy tự động") nay được gọi là "Còi hú" 97% số cửa sổ thay vì 57–63% bị gọi là báo cháy. Hai clip test đã đổi nhãn
(còi 24V H207X, còi không dây GS-WS26: tiếng quét điện tử rất lạ) vẫn khó với mọi model (0.50–0.67 có báo; phần còn lại bị đoán là
nền, không phải chuông). "DANGER có báo" ở dữ liệu gốc test thấp hơn vn2 (0.643 so với 0.690) do `smoke_alarm` gốc tụt (mục 7).

### 6.2 Mức hop trong sự kiện và mức sự kiện (kịch bản, mỗi model với config của nó)

Hop: trong mỗi sự kiện siren / smoke, các hop có p(siren) + p(smoke) ≥ 0.5; tỉ lệ hop mà lớp kia có xác suất cao hơn.
Sự kiện: cảnh báo (mới, không tính lặp) trong [onset, offset + 2 s]: "sai" = có cảnh báo lớp kia, "lật" = cả hai lớp đều báo.

| | siren val: hop sai / sự kiện sai / lật | siren VN val | siren VN test | smoke val | smoke VN val | smoke VN test |
|---|---|---|---|---|---|---|
| cu | 0.000 / 1 / 1 (của 20) | 0.000 / 0 / 0 (47) | 0.002 / 0 / 0 (46) | 0.000 / 0 / 0 | 0.018 / 0 / 0 | 0.023 / 0 / 0 |
| vn1 | 0.058 / 2 / 1 | 0.043 / 2 / 0 | 0.007 / 0 / 0 | 0.000 / 0 / 0 | 0.002 / 0 / 0 | 0.009 / 0 / 0 |
| vn2 | 0.053 / 1 / 0 | 0.010 / 1 / 0 | 0.002 / 0 / 0 | 0.000 / 0 / 0 | 0.015 / 0 / 0 | 0.022 / 0 / 0 |
| **v3** | **0.005 / 0 / 0** | **0.000 / 0 / 0** | **0.000 / 0 / 0** | 0.000 / 0 / 0 | 0.004 / 0 / 0 | 0.060 / **1** / 0 |

"Có báo DANGER" mức sự kiện (siren / smoke): VN test cu 0.80 / 0.65, vn2 0.91 / 0.72, **v3 0.89 / 0.80**; VN val cu 0.89 / 0.90, vn2
0.89 / 0.92, **v3 0.91 / 0.96**; val vn2 0.90 / 0.95, v3 0.90 / 0.95; test vn2 0.67 / 0.58, v3 0.67 / 0.50 (12 sự kiện).

Trên kịch bản tổng hợp, nhầm mức sự kiện vốn đã hiếm ở mọi model (sự kiện là clip sạch trộn vào nền ở SNR 0–20 dB). Hiện tượng
trên iPhone (âm phát lại qua loa, kéo dài, có tiếng vọng) mạnh hơn ở mức hop, nên chỉ số hop là chỉ báo tốt hơn: v3 giảm hop sai của
siren từ 5% xuống ≤ 0.5%. Ngược lại có **1/46 sự kiện chuông VN test** bị báo "Còi hú" (hop sai smoke VN test 2.2% → 6.0%).

### 6.3 Luật chống "lật" (quy tắc 10, tuỳ chọn)

Đã cài `postprocess.flip_guard_hops` / `flip_guard_margin` (`postprocess.py` quy tắc 10, có test): khi một occurrence của siren
(hoặc smoke) đang hoạt động, lớp kia chỉ được báo sau khi xác suất của nó vượt lớp đang báo hơn `margin` trong N hop liên tiếp.
Vì sau khi đổi nhãn số sự kiện "lật" của v3 là **0** trên cả sáu bộ (và số sự kiện sai chỉ 1), luật này **không được bật** trong
`soundwatch_vn_v3.yaml` và không đo thêm: không có gì để nó sửa trên dữ liệu hiện có. Nên bật thử (vd N = 3, margin 0.2) nếu thử
thật trên iPhone vẫn thấy nhảy nhãn.

## 7. So sánh cu / vn2 / v3 (mọi model chấm trên cùng dữ liệu mới)

### 7.1 Mức cửa sổ (`eval-vn`, `docs/vn_v3_reports/vn_eval_{val,test}.md`)

| Chỉ số | val: cu / vn2 / **v3** (v3 seed 7) | test: cu / vn2 / **v3** (v3 seed 7) |
|---|---|---|
| VN macro-F1 | 0.698 / 0.696 / **0.733** (0.784) | 0.722 / 0.747 / **0.769** (0.754) |
| VN recall siren | 0.860 / 0.909 / **0.934** (0.953) | 0.819 / 0.957 / 0.947 (0.953) |
| VN recall smoke_alarm | 0.762 / 0.869 / 0.857 (0.917) | 0.937 / 1.000 / 0.958 (0.968) |
| VN nền → báo động | 0.110 / 0.087 / **0.056** (0.114) | 0.118 / 0.129 / **0.112** (0.114) |
| Gốc macro-F1 | 0.801 / 0.794 / 0.763 (0.757) | 0.779 / 0.758 / 0.770 (0.778) |
| Gốc smoke_alarm (67 / 90) | 0.866 / 0.970 / **0.746** (0.821) | 0.900 / 0.844 / **0.700** (0.867) |
| Gốc car_horn (128 / 191) | 0.773 / 0.789 / 0.734 (0.664) | 0.733 / 0.775 / 0.759 (0.754) |
| Gốc siren | 0.962 / 0.968 / 0.973 | 0.561 / 0.580 / 0.580 |
| Gốc baby_cry | 0.851 / 0.881 / 0.842 | 0.815 / 0.821 / 0.810 |
| Gốc door_knock (35 / 42) | 0.771 / 0.743 / 0.743 | 0.857 / 0.762 / 0.810 |
| Gốc dog_bark | 0.754 / 0.762 / 0.741 | 0.779 / 0.760 / 0.762 |
| Gốc water_running | 0.727 / 0.701 / 0.686 | 0.713 / 0.729 / 0.697 |
| Gốc background | 0.900 / 0.897 / 0.890 | 0.881 / 0.863 / 0.883 |

So với cu, các lớp gốc của v3 tụt quá 2 điểm: **smoke_alarm** (val −12, test −20; seed 7: −4.5 / −3.3), car_horn val −3.9,
door_knock val −2.8, dog_bark val −1.3, water_running val −4.1 / test −1.6. smoke_alarm gốc bị đoán thành **nền** (test 27/90 cửa sổ),
không phải thành còi hú. Giả thuyết: lớp `smoke_alarm` mất 76 cửa sổ VN train, phần VN còn lại là bíp 3 kHz rất "sạch", nên các bíp
đồng hồ / FSD50K lẫn nền bị kéo về nền. Seed 7 ít tụt hơn nhiều, nên một phần là dao động train.

**Còi VN theo nhóm** (cửa sổ, cu / vn2 / v3):

| Nhóm | val | test |
|---|---|---|
| ambulance | 0.882 / 0.958 / **0.972** | 0.972 / 0.986 / 0.986 |
| fire | 0.867 / 0.942 / 0.946 | 0.688 / 0.983 / 0.944 |
| police | 0.812 / 0.910 / 0.875 | 0.889 / 0.981 / 0.981 |
| other | 0.917 / 0.917 / 0.917 | 0.972 / 0.972 / 0.972 |
| **relabel_v3** (còi báo cháy kiểu hú, mới) | 0.867 / 0.400 / **0.967** | 0.667 / 0.500 / 0.633 |
| passing (xe đi ngang qua) | 0.741 / 0.870 / 0.833 (54) | 0.750 / 0.958 / **1.000** (24) |
| far / mid / near | 0.825 / 0.875 / 0.891 → 0.904 / 0.943 / 0.966 | 0.904 / 0.774 / 0.795 → 0.974 / 0.914 / 0.957 |

### 7.2 Mức sự kiện (mỗi model với config của nó; `docs/vn_v3_reports/event_level_cu_vn1_vn2_v3.json`)

| Bộ | Recall tổng: cu / vn2 / **v3** / v3 street | DANGER TB | Báo nhầm/giờ trên nền | Trễ median (s) |
|---|---|---|---|---|
| test (86 sự kiện, 1 giờ nền) | 0.48 / 0.51 / **0.50** / 0.40 | 0.67 / 0.64 / 0.61 / 0.58 | 37 / 32 / **34** / 23 | 1.50 / 1.15 / 1.08 / 1.13 |
| val (138, 2 giờ) | 0.48 / 0.59 / **0.64** / 0.43 | 0.57 / 0.70 / **0.72** / 0.67 | 22 / 22 / **20** / 17 | 1.34 / 1.33 / 1.33 / 1.28 |
| VN test (92, 1 giờ) | 0.72 / 0.80 / **0.83** / 0.83 | = recall | 24 / 19 / **33** / 16 | 1.17 / 1.02 / 1.07 / 1.07 |
| VN val (95, 1 giờ) | 0.89 / 0.89 / **0.94** / 0.94 | = recall | 32 / 16 / **4** / 4 | 1.47 / 1.20 / 1.15 / 1.15 |
| DANGER ngắn test | 0.38 / 0.39 / 0.37 / 0.32 | – | – | 0.91 / 0.84 / 0.85 / 0.89 |
| DANGER ngắn val | 0.38 / 0.45 / **0.48** / 0.43 | – | – | 0.80 / 0.80 / 0.79 / 0.78 |

(vn1: test 0.49 / 35, val 0.54 / 22, VN test 0.84 / 29, VN val 0.92 / 5. vn2 street: VN test 6, VN val 4 báo nhầm/giờ.)

Recall từng lớp (cu / vn2 / **v3**):

| Bộ | smoke_alarm | car_horn | siren | baby_cry | door_knock | dog_bark | water_running |
|---|---|---|---|---|---|---|---|
| test | 0.67 / 0.50 / 0.50 | 0.67 / 0.75 / 0.67 | 0.67 / 0.67 / 0.67 | 0.23 / 0.23 / 0.23 | 0.25 / 0.50 / 0.50 | 0.46 / 0.38 / 0.38 | 0.42 / 0.58 / 0.58 |
| val | 0.70 / 0.95 / 0.95 | 0.20 / 0.30 / 0.30 | 0.80 / 0.85 / **0.90** | 0.58 / 0.68 / 0.68 | 0.15 / 0.30 / 0.35 | 0.50 / 0.55 / 0.60 | 0.42 / 0.53 / **0.68** |
| VN test | 0.63 / 0.70 / **0.76** | – | 0.80 / 0.91 / 0.89 | – | – | – | – |
| VN val | 0.90 / 0.92 / **0.96** | – | 0.89 / 0.87 / **0.91** | – | – | – | – |
| DANGER ngắn test | 0.48 / 0.48 / 0.52 | 0.23 / 0.23 / 0.19 | 0.42 / 0.45 / 0.39 | – | – | – | – |
| DANGER ngắn val | 0.53 / 0.53 / 0.59 | 0.21 / 0.30 / 0.24 | 0.41 / 0.53 / **0.62** | – | – | – | – |

Ở mức sự kiện, mọi lớp của v3 bằng hoặc cao hơn cu trên val; trên test (12–13 sự kiện mỗi lớp) smoke_alarm 0.67 → 0.50 (2 sự kiện)
và dog_bark 0.46 → 0.38 (1 sự kiện), như vn2.

Báo nhầm trên nền theo lớp (số lần; cu / vn2 / **v3** / v3 street):

| Bộ | giờ | smoke | car_horn | siren | baby | door | dog | water |
|---|---|---|---|---|---|---|---|---|
| test | 1 | 7 / 6 / 5 / 5 | 2 / 2 / 2 / 0 | 17 / 9 / **17** / 17 | 3 / 6 / 2 / 0 | 0 / 1 / 0 / 0 | 8 / 5 / 7 / 0 | 0 / 3 / 1 / 1 |
| val | 2 | 3 / 6 / 5 / 5 | 0 / 3 / 3 / 1 | 10 / 7 / **17** / 17 | 12 / 14 / 3 / 0 | 0 / 6 / 7 / 7 | 19 / 4 / 1 / 0 | 0 / 4 / 4 / 4 |
| VN test | 1 | 0 / 0 / 1 / 1 | 5 / 5 / 4 / 3 | 6 / 1 / **11** / 11 | 2 / 2 / 0 / 0 | 0 / 1 / 0 / 0 | 11 / 10 / 16 / 0 | 0 / 0 / 1 / 1 |
| VN val | 1 | 1 / 1 / 1 / 1 | 1 / 3 / 0 / 0 | 3 / 1 / 3 / 3 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 27 / 11 / 0 / 0 | 0 / 0 / 0 / 0 |

**Điểm yếu chính của v3: báo nhầm "Còi hú" trên nền** cao hơn vn2 (val 7 → 17 lần/2 giờ, VN test 1 → 11 lần/giờ; ngang cu trên
test). Nguyên nhân kép: lớp siren rộng hơn (gồm cả còi điện tử quét nhanh) và luật hai tầng 0.9 (vn2: 0.95). Trong 25 cấu hình
được chấp nhận trên val, cấu hình ít báo nhầm siren nhất vẫn có 10 + 1 lần trên val ∪ VN val (vn2: 7 + 1), đổi lại điểm val giảm
(0.771 so với 0.796). Bộ chọn hiện tại không có ngân sách báo nhầm **theo lớp**; nên thêm cho lần sau.

### 7.3 Kích thước, tài nguyên, int8

| | cu | vn2 | **v3** |
|---|---|---|---|
| Kiến trúc | DS-CNN 96→192, 4 khối | giống hệt | giống hệt (softmax, log-mel, T = 1.83 gộp vào Dense) |
| int8 / tham số / MACs | 208 KB / 152 k / 24.5 M | 208 KB / 152 k / 24.5 M | 208 KB (213 184 B) / 152 k / 24.5 M |
| Op | CONV_2D, DEPTHWISE_CONV_2D, MEAN, FULLY_CONNECTED, SOFTMAX | như cu | như cu |
| Tensor arena (ước tính) | ~117 KB | ~117 KB | 117.2 KB (đỉnh activation 93.8 KB) |
| Tụt float → int8 (macro-F1 val / test) | – | +0.05 / −0.05 | **−0.10 / −0.03** điểm (agreement 0.991 / 0.987) |
| ECE top-label val (trước → sau calibration) | 0.069 | 0.072 → 0.025 | 0.061 → 0.029; cửa sổ bão hoà 3447 → 1756 / 6073 |

## 8. Chế độ `auto` dựa vào âm nền: đánh giá

**Bộ thử** (`scripts/vn_v3/auto_mode_eval.py`): mỗi split 6 phiên × 18 phút, xen kẽ 3 phút **trong nhà** và 3 phút **ngoài
đường** (5 lần chuyển mỗi phiên, 30 lần chuyển mỗi split), ghép từ kho âm nền của split đó (không dùng train):
- trong nhà: ESC-50 các lớp sinh hoạt (gõ phím, bước chân, ho, máy giặt, hút bụi, đánh răng, …) + clip nền VN có tiêu đề trong nhà /
  quán ("Karaoke tân cổ", "Vỹ Dạ Quán"): val 166 clip (22 VN), test 161 (17 VN). **Ít âm trong nhà VN thật**, là hạn chế lớn.
- ngoài đường `vn`: clip nền VN có tiêu đề phố / chợ / giao thông (val 385, test 372); `orig`: US8K engine_idling / street_music /
  jackhammer / drilling / children_playing + ESC-50 engine (val 479, test 497).
- mức âm `rand`: mỗi clip −45 … −30 dBFS ngẫu nhiên (như `make-scenarios`; mức âm **không** mang thông tin). `plaus`: **giả định**
  trong nhà −50 … −38, ngoài đường −38 … −26 dBFS (hai khoảng tách hẳn nhau, tức thuận lợi cho RMS).

Tham số của bằng chứng âm thanh chọn **trên val** (lưới ngưỡng × trễ × hằng số thời gian × thời gian ở tối thiểu, mục tiêu
= độ chính xác − 0.01 × số lần đổi sai/giờ, chỉ dùng bằng chứng âm thanh và mức `rand`): ngưỡng −0.01, trễ 0.02, τ 30 s, ở tối
thiểu 60 s. Config v3 dùng bản **kết hợp** (bằng chứng âm thanh + RMS −35 dBFS, trễ 3 dB). Model v3, **test**:

| Luật | street VN, rand: đúng / sau 60 s / trễ median / đổi sai/giờ | street gốc, rand | street VN, plaus | street gốc, plaus |
|---|---|---|---|---|
| RMS (vn2: −35 dBFS, 30 s) | 0.54 / 0.55 / 18 s (7/30 không đổi) / 8.3 | 0.50 / 0.51 / 25 s (11/30) / 7.2 | 0.87 / 0.97 / 17 s / 2.2 | 0.89 / 1.00 / 17 s / 0.0 |
| chỉ car_horn (EMA p(horn)) | 0.56 / 0.68 / 75 s / 2.8 | 0.49 / 0.52 / 57 s / 6.1 | 0.60 / 0.70 / 59 s / 3.3 | 0.59 / 0.71 / 77 s / 3.9 |
| **âm thanh** (horn − sinh hoạt) | **0.77 / 0.91 / 30 s / 2.8** | **0.82 / 0.96 / 34 s / 2.2** | 0.79 / 0.92 / 23 s / 3.3 | 0.83 / 0.96 / 26 s / 1.7 |
| **âm thanh + RMS** (config v3) | 0.72 / 0.82 / 30 s (4/30) / 0.6 | 0.76 / 0.90 / 39 s / 1.7 | **0.83 / 0.97 / 23 s / 0.6** | **0.84 / 0.99 / 26 s / 0.0** |

(Cùng bảng với model vn2: âm thanh 0.76 / 0.84 đúng ở rand, âm thanh + RMS 0.78 / 0.86 ở plaus: cách làm không phụ thuộc model.
Chi tiết: `docs/vn_v3_reports/auto_mode_{v3,vn2}.json`.)

Đọc bảng:
- Khi mức âm không mang thông tin, RMS **không hơn đoán bừa** (50–54%, nhiều lần chuyển không bao giờ đổi). Bằng chứng âm thanh
  đúng 77–82% thời gian, 91–96% sau phút đầu mỗi đoạn; trễ đổi chế độ median 30 s (p90 ~60–80 s).
- Đếm riêng còi xe không đủ (49–60%): còi xe rõ ràng ít hơn ta nghĩ; tín hiệu tốt là xác suất **mềm**, và "âm sinh hoạt trong nhà"
  quan trọng ngang "còi xe".
- Khi mức âm tách được (giả định), RMS một mình đã rất tốt; thêm âm thanh **giảm số lần đổi sai** (2.2 → 0.6/giờ ở VN) nhưng chậm hơn
  vài giây. Ở `rand`, phần RMS trong luật kết hợp đôi khi chặn việc chuyển (4/30 lần không đổi): nếu ngưỡng dBFS sai với micro thật thì
  luật kết hợp kém hơn luật âm thanh thuần.
- **Giới hạn:** âm trong nhà phần lớn là ESC-50 (clip 5 s nhiều khoảng lặng), không có nhà ở VN thật (TV, quạt, nấu ăn, trẻ con);
  ngoài đường là clip YouTube đã được chuẩn hoá âm lượng. **Ngưỡng −35 dBFS chỉ là giá trị tạm, phải đo với micro thật** của
  đồng hồ / iPhone (trong nhà yên tĩnh, nhà có TV, quán, vỉa hè, trong xe) trước khi bật `auto` mặc định. Nếu chưa đo được, nên dùng
  luật âm thanh thuần (`street_above_dbfs: null`).

## 9. Tương thích ngược, test, công cụ

- Model / config cũ cho kết quả **giống hệt**: `scripts/vn_v3/check_compat.py` (config + 12 sự kiện demo + xác suất từng hop của
  config mặc định trùng `ios/SoundWatch/Resources/*`, không ghi file); test `test_no_display_overrides_identical_events`,
  `test_auto_rms_only_unchanged`, `test_ios_export_modes_display_and_auto`, `test_labels_header_mode_display_and_sound` (vn2 không có
  khoá mới). Số cu / vn1 / vn2 trên các bộ kịch bản mới được tính lại (khác tài liệu vn2 vì dữ liệu đổi nhãn).
- `uv run pytest`: 210 → **237** test, `uv run ruff check src tests` sạch. Test mới ở `tests/test_v3.py`.
- **Sửa công cụ:** `train-student --tag <tên>` (thí nghiệm) chỉ ghi `feature_stats_<tên>.json`; chỉ các tag mặc định
  (`distill`/`nodistill`) và `--quick` ghi `models/feature_stats.json` (`train.writes_shared_stats`, có test). `quantize` không
  `--out-dir` ghi stats của run cạnh model int8 nó viết. Trước lần chạy này `artifacts/models/feature_stats.json` đã lệch với
  `artifacts/models/student_int8.tflite` (model là vn1, stats của một thí nghiệm vn2); đã chép lại stats của vn1 cho khớp.
- `eval-vn --relabel <csv>`: nhóm các clip đã đổi sang siren thành `vehicle_type = relabel_v3`; sửa lỗi sắp xếp khi cột có ô trống.
- `scripts/vn_v2/{runs_cache,select_pp,compare}.py` đọc thư mục cache từ biến `SW_RUNS` (mặc định như cũ).

## 10. Đánh giá theo tiêu chí và đề xuất

| Tiêu chí | v3 |
|---|---|
| Hết nhảy còi hú ↔ báo cháy | ✅ cửa sổ: siren → smoke VN val 4.7% → 0.7%, gốc test 4.1% → 0.5%; còi báo cháy kiểu hú 57% → 0% bị gọi là chuông. Hop sai trong sự kiện siren 5% → ≤ 0.5%. Sự kiện lật 0. ⚠️ 1/46 sự kiện chuông VN test bị báo "Còi hú" |
| Không mất cảnh báo DANGER | ✅ VN (sự kiện: VN test smoke 0.72 → 0.80, siren 0.91 → 0.89; VN val tăng). ⚠️ gốc test cửa sổ 0.690 → 0.643 (smoke_alarm gốc) |
| Lớp khác ≤ 2 điểm so với cu | ⚠️ sự kiện: bằng hoặc hơn cu trên val; test như vn2. ❌ cửa sổ: smoke_alarm gốc −12 / −20 (seed 7: −4.5 / −3.3), water_running val −4.1, car_horn val −3.9, door_knock val −2.8 |
| Báo nhầm/giờ không tăng so với cu | ✅ test 37 → 34, val 22 → 20, VN val 32 → 4. ❌ VN test 24 → **33** (siren 11, dog_bark 16); ở `street` 16 |
| Hiển thị theo ngữ cảnh + auto theo âm nền | ✅ cài xong, tuỳ chọn, tương thích ngược; auto đúng 77–82% khi mức âm vô dụng (RMS: 50–54%) |

**Đề xuất:**
1. **Đưa v3 vào nhánh so sánh hai model trên iPhone, đặt cạnh vn2** (chưa thay model mặc định). Thử lại đúng các clip TikTok đã
   gây nhảy nhãn: với v3 cả còi xe ưu tiên lẫn còi báo cháy kiểu hú đều phải ra **"Còi hú"** (nhãn theo chế độ), chuông/bíp ra
   "Chuông / bíp báo động".
2. Ra mắt **nút Trong nhà / Ngoài đường** với nhãn theo chế độ trước. Chỉ bật `auto` sau khi đo mức âm micro thật; trong lúc chờ, nếu
   muốn thử `auto`, dùng luật âm thanh thuần (`street_above_dbfs: null`).
3. Theo dõi trên thiết bị: báo nhầm "Còi hú" ngoài đường (điểm yếu v3) và chuông/bíp gốc (đồng hồ báo thức). Lần train sau: thêm
   ngân sách báo nhầm theo lớp vào bộ chọn ngưỡng, train ≥ 2 seed và lấy trung bình / chọn theo val, thêm dữ liệu chuông/bíp thật
   (lệnh `record`) để `smoke_alarm` không chỉ dựa vào bíp 3 kHz của đầu báo khói.
4. Luật chống lật (quy tắc 10) để sẵn, chỉ bật nếu thiết bị thật vẫn nhảy nhãn.

## 11. App iOS cần sửa gì (Swift; chưa làm, `ios/SoundWatch/Resources/*` không đổi)

v3 dùng **cùng front-end log-mel, cùng softmax, cùng op, cùng kích thước tensor** như cu/vn2 (nhiệt độ đã gộp vào trọng số).
Đổi model chỉ là đổi file và ngưỡng. Phần mới nằm ở **lớp hiển thị** và **chế độ `auto`**. Các tên Swift dưới đây lấy theo nhánh
`ios-dual-model` (`Core/PostProcessor.swift`, `Core/SoundWatchConfig.swift`, `App/AppState.swift`, `UI/ContentView.swift`,
`UI/CompareView.swift`).

**Xuất tài nguyên.** `export_ios_assets.py --config config/soundwatch_vn_v3.yaml --model assets/models/vn_v3/student_int8.tflite
--stats assets/models/vn_v3/feature_stats.json`. Trên nhánh `ios-dual-model` thêm `--suffix vn3` để ghi file cạnh model cũ (bản
`--suffix` có trên nhánh đó, chưa có trên `vn-v3`; hai thay đổi không đụng nhau nên merge thẳng được). JSON v3 khác vn2 ở chỗ:
- `classes[].label_vi`: siren = "Còi hú", smoke_alarm = "Chuông / bíp báo động" (key và thứ tự lớp **không đổi**);
- `modes.modes.<mode>.label` / `.icon` / `.sf_symbol` (chỉ có khi mode có ghi đè). `sf_symbol` đã map sẵn từ tên PNG
  (`fire.png` → `flame.fill`, `siren.png` → `light.beacon.max.fill`);
- `modes.auto` có thêm `street_evidence` (lớp → trọng số), `street_score_above`, `score_hysteresis`, `evidence_window_s`,
  `min_dwell_s` (chỉ có khi dùng); `street_above_dbfs` có thể là `null` (không dùng mức âm).

**1. `SoundWatchConfig.Modes.Spec`** thêm `label: [String: String]?`, `icon: [String: String]?`,
`sfSymbol: [String: String]?` (khoá JSON `sf_symbol`), mặc định rỗng. **`Modes.Auto`** thêm `streetEvidence: [String: Double]?`,
`streetScoreAbove`, `scoreHysteresis`, `evidenceWindowS`, `minDwellS` (optional, mặc định như Python), `streetAboveDbfs` thành
`Double?`.

**2. `PostProcessor`:**
- `applyMode` tính `effectiveLabel[c]` / `effectiveSymbol[c]` từ spec của mode (nil nếu không ghi đè), như
  `PostProcessor._apply_mode` Python.
- `AlertEvent` thêm `label: String?`, `sfSymbol: String?`, gán lúc phát (cả lần lặp). Nhãn và icon "chụp" theo mode **tại
  thời điểm báo**, nên lịch sử vẫn đúng sau khi đổi mode.
- Thêm `func display(_ c: Int) -> (label: String, sfSymbol: String)` cho UI (thẻ cài đặt, banner).
- `auto` (quy tắc 9, bản v3), mỗi hop sau khi có `probs` (cả hop bị gate: probs nền một-nóng, đóng góp 0):
  ```swift
  let x = streetEvidence.reduce(0.0) { $0 + $1.value * Double(probs[classIndex[$1.key]!]) }
  let a = scoreT == nil ? 1.0 : 1 - exp(-(t - scoreT!) / evidenceWindowS)
  streetScore += a * (x - streetScore); scoreT = t
  let streetEv = streetScore >= streetScoreAbove, indoorEv = streetScore < streetScoreAbove - scoreHysteresis
  // loud / quiet từ trung vị RMS như vn2, chỉ khi streetAboveDbfs != nil
  guard t - autoT0 >= minS, lastSwitchT == nil || t - lastSwitchT! >= minDwellS else { return }
  let toStreet = hasRms ? (streetEv && !quiet) : streetEv
  let toIndoor = hasRms ? (indoorEv && !loud) : indoorEv
  ```
  Khi có bằng chứng âm thanh, `min_s` dùng nguyên giá trị (không lấy min với `window_s` như vn2). Reset `streetScore`, `scoreT`
  trong `setMode("auto")` / `reset()`; ghi `lastSwitchT` mỗi lần auto đổi mode. So khớp bằng `PostProcessor._auto_update` Python.
- Quy tắc 10 (`flip_guard_hops`) **không bật** trong `soundwatch_vn_v3.yaml` (mục 9), không cần port.

**3. UI (`ContentView` / `CompareView` / `Notifier`):**
- Mọi chỗ đang hiện `cls.labelVi` / `cls.sfSymbol` của một **cảnh báo** (thẻ cảnh báo, lịch sử, banner thông báo, màn khoá,
  log phiên `SessionRecorder`) phải dùng `event.label ?? cls.labelVi` và `event.sfSymbol ?? cls.sfSymbol`.
- Nhãn chế độ đường phố dài (40 ký tự, "Xe ưu tiên (cấp cứu / cảnh sát / cứu hoả)"): cho phép 2–3 dòng / `minimumScaleFactor`.
- Màn chính: hiện chế độ hiện tại ("Trong nhà" / "Ngoài đường", thêm "(tự động)" khi selector là `auto`). Nút chọn
  Trong nhà / Ngoài đường / Tự động gửi `setMode`. Simulator Python có sẵn bản mẫu: phím **M** đổi chế độ, dòng
  "Ngoài đường (tự động)" trên mặt đồng hồ.
- Màn cài đặt (bật/tắt lớp): vẫn dùng nhãn gốc của lớp ("Còi hú", "Chuông / bíp báo động"), vì đó là tên theo âm thanh.

**4. So sánh hai model (`ios-dual-model`):** khi so vn2 với v3, nhãn **khác nghĩa**: vn2 "siren" = còi xe cấp cứu theo nghĩa cũ,
"smoke_alarm" = báo cháy (gồm cả còi báo cháy kiểu hú). Màn so sánh nên hiện nhãn của từng model theo config của nó, và log phiên
nên ghi `class_key` + nhãn hiển thị, để khi xem lại không tưởng là v3 "nhầm" mỗi lần còi báo cháy kiểu hú được báo là "Còi hú".

**5. Kiểm thử Swift:** dựng lại `demo_expected_events_vn3.json` (sự kiện có thêm `label` khi mode có ghi đè: test nên chạy ở mode
mặc định `indoor`), thêm test: (a) đổi mode → nhãn/icon của cảnh báo sau đó đổi, cảnh báo cũ trong lịch sử giữ nguyên; (b) `auto` với
chuỗi probs có `car_horn` 0.6 mỗi 2 s → sang `street` sau vài chục giây, chuỗi `baby_cry` → về `indoor` không sớm hơn
`min_dwell_s` (như `tests/test_v3.py::test_auto_sound_evidence_switches_with_hysteresis`);
(c) config không có `label`/`icon`/trường auto mới → sự kiện và mode giống hệt hiện nay (`PostProcessorModesTests`).

**Firmware ESP32** (để tham khảo): `export --config config/soundwatch_vn_v3.yaml` sinh thêm `kModeLabelUtf8/Ascii`,
`kModeIconName[mode][class]` và `SW_AUTO_USE_SOUND`, `kAutoEvidenceWeight[class]`, `SW_AUTO_SCORE_ABOVE/HYST`,
`SW_AUTO_EVIDENCE_TAU_MS`, `SW_AUTO_MIN_DWELL_MS` trong `labels.h`; thêm 1 float + 2 timestamp RAM cho `auto`.

## 12. Việc chưa làm

- **Thử trên thiết bị thật** (clip TikTok qua loa iPhone, micro đồng hồ): ngoài phạm vi; kết quả ở đây là trên dữ liệu tổng hợp.
- **Hiệu chỉnh ngưỡng dBFS của `auto`** với micro thật; bộ thử trong nhà thiếu âm nhà ở VN thật.
- **Đo luật chống lật (quy tắc 10):** không đo vì sau khi đổi nhãn không còn sự kiện lật nào để sửa.
- **Chọn lại ngưỡng cho cu / vn2 trên dữ liệu mới:** dùng config sẵn có của chúng (như app đang chạy).
- **Giảm báo nhầm "Còi hú" và khắc phục smoke_alarm gốc:** cần ngân sách báo nhầm theo lớp và nhiều seed / dữ liệu chuông thật.
- **Không chạy `export_ios_assets.py`** (nó ghi vào `ios/SoundWatch/Resources/`), không sửa Swift. Firmware v3 sinh ở
  `artifacts/firmware_vn_v3/` (không commit).

