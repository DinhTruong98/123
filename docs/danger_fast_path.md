# Luật hai tầng cho lớp DANGER: số liệu trước/sau

Luật (xem quy tắc 4b trong `src/soundwatch/engine/postprocess.py`): một lớp DANGER đang bật, nằm ngoài cooldown và không có occurrence đang active sẽ **báo ngay** khi xác suất sau smoothing ở **một** hop đạt `postprocess.danger_fast_threshold`. Nếu bật `bg_margin` thì luật này cũng phải thoả `bg_margin`. Mọi luật khác giữ nguyên.

Đo trên cùng model (`student_int8.tflite`, 208 KB), cùng kịch bản, không train lại, không đổi ngưỡng từng lớp:
- Suy luận chạy một lần cho mỗi kịch bản, sau đó replay PostProcessor với luật tắt (`off`), 254/256 và 255/256.
- Lệnh: `uv run python -m soundwatch eval-fast-path`, kết quả ở `artifacts/reports/fast_path.json`.
- Số "off" khớp với `eval-streaming` trước khi sửa. Số 255/256 khớp với `eval-streaming` sau khi sửa.
- Ngưỡng được chọn **trên val**. Test chỉ dùng để báo cáo.

## Kịch bản thường

| | **val** off | val 254/256 | **val 255/256** | **test** off | test 254/256 | **test 255/256** |
|---|---|---|---|---|---|---|
| Recall sự kiện (tổng) | 0.512 | 0.545 | **0.545** | 0.398 | 0.441 | **0.419** |
| Recall DANGER (trung bình) | 0.55 | 0.62 | **0.62** | 0.54 | 0.64 | **0.59** |
| Recall smoke_alarm / car_horn / siren | 0.71 / 0.22 / 0.72 | 0.71 / 0.44 / 0.72 | 0.71 / 0.44 / 0.72 | 0.69 / 0.46 / 0.46 | 0.69 / 0.62 / 0.62 | 0.69 / 0.62 / 0.46 |
| Độ trễ median / p90 (s) | 1.36 / 3.46 | 1.19 / 3.48 | 1.19 / 3.48 | 1.43 / 3.23 | 1.25 / 2.95 | 1.25 / 3.03 |
| Độ trễ siren median / p90 (s) | 1.71 / 2.90 | 1.04 / 1.98 | 1.04 / 1.98 | 1.79 / 2.51 | 1.29 / 2.38 | 1.29 / 2.26 |
| **Báo nhầm/giờ, chỉ có nền** | **10.0** | 20.5 | **18.0** | **23.0** | 33.0 | **32.0** |
| Trong đó car_horn / siren / smoke (lần/giờ) | 6.0 / 0 / 0 | 14.0 / 2.5 / 0 | 12.5 / 1.5 / 0 | 0 / 10 / 6 | 0 / 20 / 6 | 0 / 19 / 6 |
| FA `outside` trong kịch bản có sự kiện (lần/giờ) | 13.0 | 18.3 | 16.8 | 27.1 | 36.5 | 35.3 |
| Sự kiện ngắn ≤ 1.5 s, DANGER (bắt được / tổng) | 2/11 | 3/11 | 3/11 | 0/5 | 1/5 | 1/5 |

Val có 123 sự kiện và 2 giờ chỉ có nền; test có 93 sự kiện và 1 giờ chỉ có nền. Báo nhầm của WARN/INFO (dog_bark, baby_cry) không đổi, vì luật mới không áp dụng cho các lớp này.

## Sự kiện DANGER ngắn (bộ kịch bản riêng)

Kịch bản thường chỉ có 5 (test) và 11 (val) sự kiện DANGER ngắn, quá ít để kết luận. Vì vậy mình sinh thêm bộ riêng:
- Các clip sự kiện DANGER (smoke_alarm, car_horn, siren) của từng split được cắt thành đoạn ngẫu nhiên dài đúng 0.5 / 1.0 / 1.5 s, có fade 10 ms, rồi trộn vào nền như `make-scenarios` (SNR 0/5/10/20 dB, sự kiện cách nhau ít nhất 4 s). Không có kịch bản chỉ có nền.
- Mỗi split có 20 kịch bản × 120 s.
- Lệnh:

```bash
uv run python -m soundwatch make-scenarios --split test --classes danger --event-lengths 0.5,1,1.5 \
    --n-events 20 --n-background 0 --duration 120 --seed 7 --out artifacts/data/scenarios_short_test
# tương tự với --split val --out artifacts/data/scenarios_short_val
```

| | **val** off | val 254/256 | **val 255/256** | **test** off | test 254/256 | **test 255/256** |
|---|---|---|---|---|---|---|
| Recall (tổng, toàn DANGER) | 0.32 | 0.50 | **0.50** | 0.27 | 0.44 | **0.38** |
| Recall smoke_alarm / car_horn / siren | 0.39 / 0.18 / 0.39 | 0.57 / 0.36 / 0.57 | 0.57 / 0.36 / 0.57 | 0.48 / 0.27 / 0.07 | 0.58 / 0.50 / 0.23 | 0.58 / 0.40 / 0.17 |
| Bắt được theo độ dài 0.5 s / 1 s / 1.5 s | 5/34 · 8/25 · 14/25 | 14/34 · 12/25 · 16/25 | 14/34 · 12/25 · 16/25 | 2/32 · 9/27 · 14/32 | 8/32 · 12/27 · 20/32 | 7/32 · 9/27 · 19/32 |
| Độ trễ median / p90 (s) | 1.23 / 1.95 | 0.86 / 1.50 | 0.92 / 1.50 | 1.21 / 1.57 | 0.84 / 1.32 | 0.81 / 1.49 |
| FA `outside` (lần/giờ) | 8.4 | 18.4 | 16.7 | 22.0 | 30.5 | 27.1 |

## Demo (`make demo-headless`)

Vẫn bắt được 6/7 âm và không báo nhầm. Cả ba lớp DANGER báo **sớm hơn 0.5 s**: smoke_alarm 3.5 → 3.0 s, car_horn 12.5 → 12.0 s, siren 22.5 → 22.0 s. baby_cry, door_knock và water_running không đổi. dog_bark vẫn bị sót như trước. `ios/SoundWatch/Resources/demo_expected_events.json` đã được tạo lại theo kết quả này.

## Kết luận

- 255/256 được chọn trên val. So với 254/256, recall trên val **giống hệt** nhưng báo nhầm ít hơn (18.0 so với 20.5 lần/giờ). 254/256 chỉ tốt hơn trên test, mà test không được dùng để chọn.
- **Lợi ích (val):**
  - Recall DANGER trung bình +7 điểm (0.55 → 0.62).
  - Recall sự kiện DANGER ngắn +18 điểm (0.32 → 0.50). Với đoạn 0.5 s: 5 → 14/34.
  - Độ trễ siren median −0.67 s (1.71 → 1.04 s).
- **Cái giá:** báo nhầm trên nền tăng khoảng **+8 lần/giờ ở val** (10 → 18, chủ yếu car_horn) và **+9 lần/giờ ở test** (23 → 32, chủ yếu siren). Tính ra thêm khoảng 1 lần báo nhầm DANGER mỗi 7 phút.
- **Đề xuất:** giữ luật ở **255/256**. Đã bật trong `config/soundwatch.yaml`; xoá dòng đó hoặc đặt `null` để tắt. Mức báo nhầm chấp nhận được là quyết định sản phẩm, để Dinh chốt dựa trên các con số trên. Muốn giảm báo nhầm thì cần sửa model hoặc dữ liệu (hard negative cho siren và car_horn, xem `docs/vn_siren_finetune.md`), hạ ngưỡng nữa không giải quyết được.
