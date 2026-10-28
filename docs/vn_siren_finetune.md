# Kế hoạch fine-tune còi xe ưu tiên Việt Nam

Đây mới là kế hoạch, chưa train. Bối cảnh: model hiện tại (`assets/models/student_int8.tflite`) chỉ học còi nước ngoài, từ UrbanSound8K, ESC-50 và FSD50K. Còi xe cấp cứu, xe cảnh sát và xe cứu hoả ở Việt Nam có nhịp và cao độ khác, nên dễ bị bỏ sót hoặc đoán thành âm nền. Ngoài ra `siren` vốn là lớp báo nhầm nhiều nhất: nhạc, máy khoan, động cơ hay bị đoán thành còi. Xem `artifacts/reports/fa_analysis.md`.

## 1. Gộp vào `siren` hay tách thành lớp riêng

| | Gộp vào `siren` (khuyến nghị) | Tách lớp mới, ví dụ `siren_vn` |
|---|---|---|
| Ý nghĩa với người dùng | Cùng mức DANGER, cùng hành động (nhường đường). Chỉ đổi nhãn thành "Còi cấp cứu / cảnh sát" | Biết rõ loại xe, nhưng người dùng không cần phân biệt |
| Model | Giữ 8 output. Chỉ fine-tune trọng số, kiến trúc và contract `(49,40,1) → 8` không đổi | 9 output: đổi `Dense` cuối, train lại head teacher (9 lớp), tính lại `teacher_logits_*` |
| Dữ liệu | Còi VN bổ sung vào lớp có sẵn. Hai nhóm dữ liệu giúp nhau | Lớp mới ít dữ liệu, dễ lẫn với `siren` (hai lớp gần giống nhau, nên khó phân tách) |
| App iOS | Chỉ đổi `label_vi` trong `config/soundwatch.yaml`, rồi chạy lại `ios/scripts/export_ios_assets.py` | Thêm lớp vào config (icon, màu, mẫu rung), thêm `SF_SYMBOLS` trong script export, sửa các test giả định 8 lớp (`tests/test_config.py`, test Swift), xuất lại `demo_expected_events.json` |
| ESP32 | `labels.h` chỉ đổi chuỗi. Kích thước và arena không đổi | `labels.h` thêm 1 phần tử. Model thêm khoảng 200 tham số (vài trăm byte). Arena gần như không đổi |

Chọn **gộp**. Chỉ tách khi có lý do sản phẩm thật sự cần phân biệt loại xe.

## 2. Nguồn dữ liệu

- **Tự thu (chính):** dùng lệnh có sẵn
  `uv run python -m soundwatch record --class siren --seconds 10 --count 20`. File WAV 16 kHz được ghi vào `artifacts/data/custom/<lớp>/`. Có thể thu bằng app iPhone, xem mục 5.
  - Cần thu đủ các tình huống: xe đứng yên, xe **chạy ngang qua** (hiệu ứng Doppler, xa → gần → xa), xe ở xa hoặc bị nhà cửa che khuất, trong xe hay trong nhà nghe ra ngoài, lúc mưa, giờ cao điểm.
  - Thu nhiều loại xe: cấp cứu (115 và bệnh viện tư), cảnh sát, cứu hoả, và còi "ưu tiên" lắp trên xe đoàn.
- **Lưu ý: `build-dataset` hiện chưa đọc `artifacts/data/custom/`.** Cần viết thêm `collect_custom` trong `src/soundwatch/data/build_dataset.py`. Hàm này đọc metadata theo buổi thu (mục 3) để chia split, và đưa clip test vào `artifacts/data/events/<split>/<lớp>/`. `make-scenarios` có sẵn tham số `noise_dir` và `events_dir` trong `load_library`, nhưng CLI chưa có cờ tương ứng. Cần thêm `--noise-dir` và `--events-dir` để sinh kịch bản từ dữ liệu VN.
- **Nguồn trên mạng (phụ, cẩn thận giấy phép):**
  - **YouTube:** điều khoản dịch vụ cấm tải xuống khi không có nút tải của YouTube hoặc không được chủ kênh cho phép. Bản quyền thuộc người đăng. Chỉ dùng khi xin được phép. Không đưa clip vào repo và không phát hành model train từ nguồn này nếu chưa rõ quyền. AudioSet cũng chỉ là danh sách ID YouTube, nên vướng cùng vấn đề.
  - **Freesound:** giấy phép ghi theo từng clip (CC0, CC-BY, CC-BY-NC). Ghi lại tác giả và giấy phép từng clip trong `SOURCES.csv`. Clip CC-BY-NC thì không dùng cho mục đích thương mại, giống ESC-50 và UrbanSound8K.
  - **Dữ liệu tự thu ngoài đường:** sẽ lẫn tiếng người nói. Giữ dữ liệu ở chế độ riêng tư, không công bố audio thô. Tiếng nói có thể bị coi là dữ liệu cá nhân, xem Nghị định 13/2023/NĐ-CP. Nếu cần chia sẻ, chỉ chia sẻ đặc trưng log-mel hoặc model.

## 3. Số lượng và chia tập

Mục tiêu tối thiểu, có thể tăng dần:

| Loại | Tối thiểu | Tốt |
|---|---|---|
| Clip còi VN (mỗi clip 5–15 s, tính phần còi thật sự kêu) | 150 clip, khoảng 25 phút | 400 clip trở lên, khoảng 60 phút |
| Trong đó: xe chạy ngang qua (Doppler) | 50 | 150 |
| Số buổi thu / số địa điểm | 15 / 8 | 40 / 20 |
| Âm nền đường phố VN, **không có còi** (xe máy, bấm còi, karaoke, nhạc quán, công trình, chợ, mưa) | 3 giờ | 10 giờ trở lên |
| Âm nền trong nhà VN (TV, quạt, máy giặt, bếp) | 1 giờ | 3 giờ |

- **Chia theo buổi thu, không theo cửa sổ 1 s.** Mỗi buổi là một địa điểm trong một khoảng thời gian liên tục. Mọi clip của một buổi nằm trọn trong một split. Gợi ý 70 / 15 / 15 theo số buổi, và địa điểm của test không xuất hiện ở train.
  - Lý do: cắt theo cửa sổ thì cùng một chiếc xe hoặc cùng một ngã tư sẽ có mặt ở cả train lẫn test, làm kết quả đẹp giả.
- **Tập test "còi VN thật" giữ riêng:** khoảng 30 clip trở lên, cộng 1 giờ âm nền VN, từ các buổi và địa điểm riêng. Nên thu bằng một máy khác nếu được. Tập này không dùng để tune ngưỡng hay chọn epoch, chỉ dùng để báo cáo cuối.
- Lớp `car_horn` cũng bị ảnh hưởng: ở VN tiếng bấm còi xe máy và ô tô rất dày. Âm nền VN có còi xe thì **phải gán nhãn `car_horn`** (hoặc cắt bỏ đoạn đó), không được tính là âm nền, nếu không model sẽ học sai.

## 4. Quy trình fine-tune

1. **Dữ liệu:** `download` và `build-dataset` như cũ, thêm `--with-custom` (cần viết, mục 2). Tạo bộ kịch bản VN: test và val, có sự kiện ngắn kiểu `make-scenarios --event-lengths 0.5,1,1.5` như phần 1, và kịch bản chỉ có nền VN.
2. **Teacher:** train lại head YAMNet bằng `train-teacher` trên dữ liệu đã trộn, rồi tính lại `teacher_logits_*`. Kiểm tra recall `siren` của teacher trên val VN trước khi distill.
3. **Student:** fine-tune từ `artifacts/models/student_distill.keras` (output logits; `student_float.keras` là bản đã có softmax, chỉ dùng để quantize).
   - Learning rate nhỏ, khoảng 1e-4 đến 3e-5. Chạy 10–20 epoch, early stopping theo macro-F1 trên val (cả val cũ lẫn val VN).
   - Nên giữ distill (α = 0.5, T = 4). Nếu head teacher vẫn kém trên còi VN thì giảm α hoặc tắt distill cho mẫu VN.
   - **Trộn dữ liệu cũ** để không quên các lớp khác: mỗi epoch lấy toàn bộ mẫu VN (oversample cho tới khi chiếm khoảng 25–30% số mẫu `siren`), và giữ cap theo lớp như hiện tại.
   - Đưa âm nền VN vào làm **hard negative**, và trộn còi VN với nền VN ở SNR 0–20 dB.
4. **Quantize và export:** chạy `quantize`, `export`, rồi `uv run python ios/scripts/export_ios_assets.py`. Kiểm tra int8 không tụt quá 2 điểm.
5. **Đo:** dùng đúng bộ chỉ số của phần 1 (`eval-fast-path`, `eval-streaming` trên test và val cũ), cộng thêm tập còi VN:
   - recall còi VN: tổng, theo khoảng cách, xe đứng yên so với chạy ngang qua, sự kiện ngắn;
   - độ trễ;
   - báo nhầm/giờ trên nền VN, tách theo lớp.

   Cần **không làm tụt** recall các lớp khác quá 2 điểm, và không tăng báo nhầm/giờ trên test cũ.
6. **Ngưỡng:** chạy lại `tune` (smoothing, k-of-n, ngưỡng từng lớp) và chọn lại `danger_fast_threshold` **chỉ trên val**. Test (cũ và VN) chỉ dùng để báo cáo.

Rủi ro chính là báo nhầm `siren` tăng, do nhạc karaoke và loa kẹo kéo có tiếng giống còi hú. Để giảm rủi ro: thu nhiều âm nền loại này, đánh dấu báo nhầm ngay trên app (mục 5), và giữ ngưỡng chọn trên val.

## 5. App iPhone: chế độ "thu clip có gắn nhãn"

Nên làm, vì đây là cách nhanh nhất để có dữ liệu VN thật, thu bằng đúng micro sẽ chạy model. Yêu cầu cho session trên Mac:

- **Giao diện:** chọn lớp (`siren`, `car_horn`, `background`…, mặc định lấy từ config) và nút **"Lưu 10 s vừa rồi"**. Ứng dụng luôn giữ một ring buffer khoảng 15 s, vì xe chạy ngang qua thường đã đi xa khi kịp bấm. Thêm nút thu 5–10 s tính từ lúc bấm. Thêm nút **"Báo nhầm"** trên màn hình cảnh báo: lưu đoạn audio gây báo nhầm, nhãn mặc định `background`, kèm lớp model đã đoán, để dùng làm hard negative.
- **Âm thanh:** WAV PCM 16-bit mono **16 kHz**, lấy sau đúng bước resample mà engine dùng. AVAudioSession mode `.measurement`, giống lúc suy luận. Có thể lưu kèm bản gốc 48 kHz nếu còn dung lượng.
- **Tên file:** `<class_key>__<session_id>__<YYYYMMDDTHHMMSS>__<seq>.wav`, ví dụ `siren__s20261101-hn-caugiay__20261101T081512__003.wav`. `session_id` do người dùng bấm "bắt đầu buổi thu" tạo ra, gồm ngày và tên địa điểm ngắn.
- **Metadata:** một file `.json` cùng tên đặt cạnh mỗi clip, gồm:
  - `class_key`, `session_id`, `started_at` (ISO 8601, có múi giờ), `duration_s`, `sample_rate`;
  - `device_model`, `ios_version`, `app_version`, `model_sha256`;
  - `location`: lat/lon **làm tròn khoảng 100 m**, chỉ ghi khi người dùng bật. Kèm `place_label` tự gõ;
  - `scene` (`street` / `indoor` / `vehicle`), `distance` (`near` / `mid` / `far`), `passing` (bool), `vehicle_type` (`ambulance` / `police` / `fire` / `other`), `notes`;
  - `predictions`: probs của model ở các hop trong clip, để tìm các trường hợp khó;
  - `trigger`: `manual` / `false_alarm`.
- **Xuất dữ liệu:** "Xuất buổi thu", nén zip toàn bộ clip, JSON và `session.json`, rồi gửi qua share sheet (Files, AirDrop). Phía repo viết `ios/scripts/import_ios_clips.py`: giải nén vào `artifacts/data/custom/<class_key>/`, kiểm tra 16 kHz mono, và gộp metadata thành `artifacts/data/custom/sessions.csv` để `build-dataset --with-custom` chia split theo `session_id`.
- **Quyền riêng tư:** clip chỉ nằm trong app cho tới khi người dùng chủ động xuất. Vị trí là tuỳ chọn. Hiện một dòng nhắc rằng clip có thể chứa tiếng người xung quanh.
