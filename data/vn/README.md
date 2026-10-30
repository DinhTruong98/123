# vn-sound-data: âm thanh Việt Nam để fine-tune SoundWatch

> Trong repo chỉ có `manifest.csv` (danh sách clip, không có audio). Tạo lại audio bằng
> `uv run --with yt-dlp python scripts/rebuild_vn_data.py --out ../vn-sound-data`. Phần "Tái tạo" bên dưới mô tả
> pipeline tìm và gán nhãn gốc (script nằm ngoài repo, không cần chạy lại).

Bộ dữ liệu bổ sung cho kế hoạch `docs/vn_siren_finetune.md` (nhánh `danger-fast-path` của repo `soundwatch`). Model hiện
tại chỉ học còi nước ngoài, nên bộ này gom còi xe ưu tiên, chuông/còi báo cháy và âm nền **ở Việt Nam**, lấy từ
YouTube và Freesound. Quyết định đã chốt: còi VN gộp vào lớp `siren`, chuông báo cháy VN gộp vào `smoke_alarm`, không
tạo lớp mới.

> **Không commit audio lên GitHub, không phát hành lại.** Xem [`SOURCES.md`](SOURCES.md) (nguồn và giấy phép từng clip).

## Số liệu (2026-10-28, bản 2)

| Lớp | train | val | test | Tổng | Nguồn / kênh | Clip CC |
|---|---|---|---|---|---|---|
| `siren` | 442 (73 ph) | 99 (16 ph) | 97 (16 ph) | **638 (105 ph)** | 162 / 97 | 65 |
| `smoke_alarm` | 110 (16 ph) | 24 (3 ph) | 24 (3 ph) | **158 (23 ph)** | 53 / 47 | 19 |
| `background` | 2039 (5,7 h) | 437 (73 ph) | 437 (73 ph) | **2913 (8,1 h)** | 130 / 98 | 518 |
| `car_horn` | 26 (2 ph) | 5 (25 s) | 5 (25 s) | 36 (3 ph) | 21 / 19 | 8 |

- `siren`: loại xe fire 268, ambulance 176, police 161, other 33; cảnh street 550, bản thu sạch 64, vehicle 24.
- **43 clip xe đi ngang qua** (train/val/test: 28/9/6; police 22, fire 12, other 5, ambulance 4).
- `label_confidence`: siren high 585 / medium 12 / low 41; smoke_alarm 133 / 5 / 20; background high 2738 / medium 175.
- Tổng 342 nguồn (238 kênh/tác giả), 610/3745 clip có giấy phép CC.
- `background` gồm đường phố, chợ, quán cà phê, mưa, công trình, karaoke/loa kẹo kéo, rao hàng, kèn đám ma, trong nhà
  (quạt, máy giặt) và 5 bản nhạc phòng thu (karaoke, nhạc đám cưới); vượt mục tiêu 3 giờ.

### Split × tầng

| Tầng | train | val | test | Tổng |
|---|---|---|---|---|
| `background` | 2039 (70%) | 437 (15%) | 437 (15%) | 2913 |
| `car_horn` | 26 (72%) | 5 (14%) | 5 (14%) | 36 |
| `siren/ambulance` | 120 (70%) | 26 (15%) | 26 (15%) | 172 |
| `siren/ambulance/passing` | 3 (75%) | 1 (25%) | 0 (0%) | 4 |
| `siren/fire` | 180 (70%) | 38 (15%) | 38 (15%) | 256 |
| `siren/fire/passing` | 7 (58%) | 3 (25%) | 2 (17%) | 12 |
| `siren/other` | 17 (61%) | 5 (18%) | 6 (21%) | 28 |
| `siren/other/passing` | 3 (60%) | 2 (40%) | 0 (0%) | 5 |
| `siren/police` | 97 (70%) | 21 (15%) | 21 (15%) | 139 |
| `siren/police/passing` | 15 (68%) | 3 (14%) | 4 (18%) | 22 |
| `smoke_alarm` | 110 (70%) | 24 (15%) | 24 (15%) | 158 |

Kiểm tra rò rỉ: 0/342 nguồn và 0/238 kênh/tác giả nằm ở hơn một split.

## Cấu trúc

```
vn-sound-data/
├── siren/          còi cứu thương, cảnh sát/CSGT dẫn đoàn, cứu hoả (VN)
├── smoke_alarm/    chuông báo cháy, còi-đèn báo cháy, đầu báo khói (thiết bị dùng ở VN)
├── background/     âm nền VN KHÔNG có còi/chuông/bấm còi (hard negative)
├── car_horn/       tiếng bấm còi xe tách ra từ âm nền (phụ, ít)
├── manifest.csv    mọi clip + nhãn + split
├── listening_check.md            danh sách nghe kiểm tra (20 clip ngẫu nhiên mỗi lớp)
├── listening_check_<lớp>.m3u     cùng danh sách, dạng playlist (VLC/IINA)
├── SOURCES.md      nguồn, giấy phép, cảnh báo
├── tools/          script tái tạo bộ dữ liệu (xem cuối file)
└── _raw/           audio thô đã tải (16 kHz mono), chỉ để cắt lại; có thể xoá
```

Mỗi clip: WAV PCM 16-bit mono 16 kHz, tên `<class_key>__<source_id>__<start_s>__<seq>.wav`, `source_id` là `yt-<id>`
hoặc `fs-<id>`, `start_s` là giây trong video/bản thu gốc. Kèm file `.json` cùng tên:

| Trường | Ý nghĩa |
|---|---|
| `class_key` | `siren` / `smoke_alarm` / `background` / `car_horn` (khớp key trong `config/soundwatch.yaml`) |
| `source_url`, `platform`, `uploader`, `channel`, `license` | nguồn; URL YouTube có `&t=` trỏ đúng giây bắt đầu |
| `start_s`, `end_s`, `duration_s` | vị trí trong nguồn gốc |
| `vehicle_type` | `ambulance` / `police` / `fire` / `other`, suy từ từ khoá trong tiêu đề (chỉ xét mô tả khi tiêu đề không có từ khoá); nhiều loại thì lấy từ khoá xuất hiện trước. Đoạn còi lấy từ video diễn tập PCCC gán `fire` (siren) |
| `passing` | xe đi ngang qua micro (xem "Cách gán nhãn") |
| `scene` | `street` / `indoor` / `vehicle` (trong xe ưu tiên); `null` với bản thu "sạch" (âm mẫu, demo thiết bị) |
| `distance` | `near` / `mid` / `far`, **tương đối**: tam phân vị mức âm dải 0,5–2,5 kHz trong lớp (audio YouTube đã chuẩn hoá âm lượng nên không có ngưỡng tuyệt đối); chỉ để tham khảo |
| `label_confidence` | `high` / `medium` / `low` |
| `split`, `group_key` | split và nhóm dùng để chia (kênh YouTube hoặc tác giả Freesound) |
| `yamnet` | điểm YAMNet trung bình trong clip (siren, alarm, horn, music, speech) |
| `notes` | ghi chú (heuristic dùng, bản nhạc phòng thu…) |

`manifest.csv` có cùng thông tin, một dòng một clip, cộng cột `license_cc` (True nếu Creative Commons).

## Cách gán nhãn

1. **Chọn nguồn theo tiêu đề** (đọc tay): bỏ đồ chơi, hoạt hình, nhạc chuông, game, video nước ngoài, video tổng hợp
   "khắp thế giới". Bỏ chuông báo giờ trường học (cùng loại chuông điện nhưng không phải báo cháy, để tránh dạy model
   báo động nhầm).
2. **Lọc sơ bộ bằng YAMNet** (tflite, AudioSet 521 lớp, hop 0,48 s). Còi VN vẫn là còi hú nên YAMNet bắt tốt (điểm
   0,3–0,9). Model SoundWatch **không** được dùng để gán nhãn.
   - `siren`: vùng có điểm nhóm còi (Siren, Emergency vehicle, Police/Ambulance/Fire engine…) trung bình trượt ≥ 0,2,
     cắt thành clip 5–15 s. Tối đa 6 clip thường + 4 clip "đi ngang qua" mỗi video, 30 clip mỗi kênh.
   - `smoke_alarm`: như trên với max(nhóm báo động, nhóm còi), chỉ trên video chuông/còi báo cháy. Với video diễn tập
     PCCC, đoạn thiên về còi xe chuyển sang `siren` và phải duyệt bằng mắt.
   - `background`: cửa sổ 10 s mà điểm còi < 0,1, báo động < 0,15, bấm còi xe < 0,25 suốt cửa sổ; tối đa 6 phút mỗi
     video YouTube, 5 phút mỗi bản Freesound, 3 phút mỗi bản nhạc. Đoạn có bấm còi xe rõ (≥ 0,5) cắt thành clip
     `car_horn` 5 s thay vì để lẫn vào nền.
3. **Duyệt bằng mắt** trên spectrogram 0–4 kHz (script `sheets.py`): mọi clip `siren`/`smoke_alarm` điểm < 0,35 và
   mọi clip từ video diễn tập phải được xem và chấp nhận mới vào bộ dữ liệu; nhóm điểm ≥ 0,35 được xem theo mẫu. Đã
   loại: đoạn intro ghép nhiều tiếng còi của kênh (TGG, XE ƯU TIÊN), video có nhạc nền chèn thêm, đoạn bị cắt ghép, đoạn còi xe máy/tiếng nói bị
   nhận nhầm.
4. **`label_confidence`**: `high` khi điểm YAMNet ≥ 0,35 và không có nhạc; `medium` 0,2–0,35; `low` khi đã xem và
   thấy còi/chuông nhưng mờ hoặc lẫn tiếng khác. Âm nền: `high` khi gần như không có điểm còi/chuông/bấm còi.
5. **`passing` (xe đi ngang qua)**: heuristic tìm đỉnh mức âm của thành phần còi cao hơn mức thấp nhất trong 5 s
   trước **và** 5 s sau ≥ 7 dB, rồi xác nhận bằng mắt (khoảng 1/5 ứng viên được giữ). **Không đo Doppler tự động**: đã thử (tương quan phổ, bám
   tần số cơ bản) nhưng không ổn định vì còi VN quét tần số liên tục và xe thường chạy chậm giữa dòng xe máy.
6. **Khử trùng lặp**: cùng một đoạn phim đăng lại ở nhiều kênh/video (rất hay gặp với tin tức) được phát hiện bằng
   tương quan log-mel (trượt ±3 s); giữ một bản, ưu tiên bản có giấy phép CC.

## Chia split

**Stratified group split** (`tools/finalize.py`, hàm `assign_splits`):

- Nhóm = kênh YouTube (hoặc tác giả Freesound). Mọi clip của một nhóm nằm trọn trong một split, nên cũng trọn video.
- Tầng = lớp; riêng `siren` tách theo `vehicle_type` × `passing`. Mục tiêu mỗi tầng 70/15/15.
- Gán tham lam (nhóm lớn trước), rồi tìm kiếm cục bộ 20 000 bước (dời một nhóm hoặc đổi chỗ hai nhóm), giảm tổng độ
  lệch bình phương có trọng số (tầng nhỏ như passing, police có trọng số cao hơn). Seed cố định, chạy lại ra cùng kết quả.
- Không cần ngoại lệ nhóm theo video: kênh chiếm nhiều nhất một loại xe là "Nhật Ký Dẫn Đoàn" (khoảng 1/5 clip police),
  vẫn chia được theo kênh.
- Báo cáo rò rỉ in mỗi lần chạy (`tools/work/split_report.md`): hiện 0 nguồn, 0 kênh nằm ở hơn một split.
- Tầng passing quá nhỏ nên lệch: ambulance/passing chỉ có 4 clip, test không có clip nào.

## Dùng để fine-tune (ở môi trường gốc)

Kế hoạch (mục 2, 5) dự kiến dữ liệu tự thu nằm ở `artifacts/data/custom/<lớp>/` và chia split theo `session_id` trong
`sessions.csv`. Với bộ này:

1. Chép nguyên `vn-sound-data/` (trừ `_raw/`, `tools/.venv/`) sang máy gốc, **không qua GitHub**:
   ```bash
   cd 03-embedded-ai/project
   tar --exclude='vn-sound-data/_raw' --exclude='vn-sound-data/tools/.venv' --exclude='vn-sound-data/tools/work' \
       -czf vn-sound-data.tgz vn-sound-data
   ```
   rồi chuyển bằng AirDrop / ổ USB / `scp` / Google Drive riêng tư, giải nén vào `soundwatch/artifacts/data/custom/`.
2. `collect_custom` (cần viết, mục 2 kế hoạch) đọc `manifest.csv`: dùng cột `split` có sẵn (hoặc `group_key` thay cho
   `session_id`), cột `class_key` làm nhãn. Clip `test` để riêng làm tập "còi VN thật", không dùng tune ngưỡng.
3. Có thể lọc theo `label_confidence` (ví dụ bỏ `low` khi train, giữ để phân tích lỗi) và dùng cờ `passing`,
   `distance`, `vehicle_type` để báo recall theo nhóm như mục 5 kế hoạch.
4. `background` dùng làm hard negative và làm nền để trộn còi ở SNR 0–20 dB. Phần nhạc (karaoke, nhạc đám cưới) là bản
   phòng thu, nên vẫn cần thêm karaoke/loa kẹo kéo thu thật.

## Thu hẹp khoảng cách miền (gợi ý, chưa làm)

Clip lấy từ video đã qua nén (AAC/Opus), micro điện thoại/camera khác nhau, có khi đã chỉnh âm lượng. Model chạy trên
micro iPhone, nên nên **phát lại clip qua loa và thu bằng iPhone** (đúng app SoundWatch ở chế độ thu clip, hoặc Voice
Memos rồi resample 16 kHz):

- Phát `siren/` và `smoke_alarm/` qua loa máy tính hoặc loa Bluetooth trong phòng, iPhone đặt cách 1 m, 3 m, khác
  phòng, cửa đóng/mở; mỗi file thu lại giữ cùng tên, thêm hậu tố `__replay-<vị trí>`.
- Âm lượng loa vừa phải (không méo). Thu kèm vài phút phòng yên tĩnh làm `background`.
- Clip thu lại kế thừa `split` của clip gốc (không để bản gốc ở train mà bản thu lại ở test).
- Dinh làm được ở nhà, không cần ra đường.

## Hạn chế đã biết

- **`passing` mới có 43 clip** (mục tiêu ≥ 50), ambulance chỉ 4. Video trên mạng hay cắt ngay khi xe vừa qua, xe ưu tiên ở VN thường
  chạy chậm giữa dòng xe máy, và heuristic chỉ đúng khoảng 1/4 nên phần lớn đề xuất đã bị loại khi duyệt.
- `smoke_alarm` 158 clip nhưng chỉ 23 phút, phần lớn từ video demo thiết bị. Chuông điện báo giờ trường học bị loại
  có chủ ý.
- `car_horn` chỉ là phần phụ (36 clip tách từ âm nền), chưa đủ để fine-tune lớp này.
- Mới có khoảng 15% clip có giấy phép CC; phần còn lại là YouTube thường, chỉ dùng nội bộ.

- Phần lớn clip `siren` ở TP.HCM (kênh TGG, FirePoAm, XE ƯU TIÊN, KL Emergency), ít Hà Nội và tỉnh.
- Video tin tức thường có lời dẫn đè lên còi; clip có điểm `speech` cao vẫn giữ nếu còi rõ.
- `vehicle_type` suy từ tiêu đề, video có nhiều loại xe sẽ gán theo từ khoá đầu tiên tìm được.
- `smoke_alarm` phần lớn là video demo thiết bị (thu gần, trong phòng) và diễn tập; ít cảnh chuông kêu thật trong chung
  cư nghe từ trong căn hộ.
- YouTube chặn tạm thời theo IP khi tải nhiều; video đòi đăng nhập đã bị bỏ qua, không dùng cookie.

## Tái tạo

```bash
cd vn-sound-data/tools
uv venv -p 3.12 .venv && uv pip install -p .venv numpy scipy matplotlib soundfile ai-edge-litert
# YAMNet tflite + class map vào tools/yamnet/ (URL trong scan.py / README này)
.venv/bin/python -I search.py        # tìm ứng viên YouTube -> work/candidates.jsonl
.venv/bin/python -I fs_search.py     # tìm Freesound -> work/fs_candidates.jsonl
#   (chọn tay id vào work/selected.txt)
.venv/bin/python -I download.py yt && .venv/bin/python -I download.py fs   # -> _raw/
.venv/bin/python -I scan.py          # YAMNet -> work/scan/
.venv/bin/python -I propose.py       # đề xuất clip -> work/proposals.csv
.venv/bin/python -I sheets.py siren all   # spectrogram để duyệt -> work/review/
.venv/bin/python -I decide.py ...    # ghi quyết định duyệt -> work/decisions.json
.venv/bin/python -I finalize.py      # cắt clip, JSON, manifest, split, SOURCES.md, danh sách nghe
.venv/bin/python -I listening.py     # listening_check.md
```

YAMNet: `https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite`,
class map: `https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv`.
