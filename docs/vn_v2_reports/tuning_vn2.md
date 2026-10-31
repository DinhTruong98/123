# Tinh chỉnh hậu xử lý (WP8) – kịch bản **val**

Model `assets/models/vn_v2/student_int8.tflite`; kịch bản `artifacts/data/scenarios_val`: 30 kịch bản có sự kiện (130 sự kiện), 20 kịch bản chỉ có nền (2.00 giờ). Mục tiêu: tối đa recall có trọng số (DANGER ×2) với báo nhầm nền ≤ ngân sách và trễ median ≤ 1.5 s. Ngưỡng từng lớp chọn tham lam trên lưới 0.3–0.996094. Sinh bởi `python -m soundwatch tune`.

## Cấu hình tốt nhất theo ngân sách báo nhầm

| Ngân sách FA/h | Hậu xử lý | Recall (trọng số) | Recall | Recall DANGER TB / min | FA/h nền | Trễ median / p90 (s) | Ngưỡng |
|---|---|---|---|---|---|---|---|
| ≤ 10 | smooth=1 k/n=2/3 margin=– | 0.535 | 0.492 | 0.632 / 0.316 | 10.0 | 1.37 / 5.06 | smoke_alarm=0.85, car_horn=0.80, siren=1.00, baby_cry=0.99, door_knock=0.45, dog_bark=1.00, water_running=0.97 |
| ≤ 15 | smooth=1 k/n=1/1 margin=– | 0.556 | 0.531 | 0.614 / 0.263 | 15.0 | 1.17 / 4.21 | smoke_alarm=0.85, car_horn=1.00, siren=1.00, baby_cry=1.00, door_knock=0.55, dog_bark=1.00, water_running=1.00 |
| ≤ 22 | smooth=1 k/n=1/1 margin=– | 0.583 | 0.554 | 0.649 / 0.316 | 22.0 | 0.99 / 4.09 | smoke_alarm=0.75, car_horn=0.85, siren=1.00, baby_cry=1.00, door_knock=0.40, dog_bark=1.00, water_running=1.00 |
## Pareto front (recall có trọng số ↑, FA/h ↓, trễ median ↓)
Chỉ các điểm khả thi với ngân sách của chúng; ✓ = trễ median ≤ 1.5 s.
| Hậu xử lý | Ngân sách | Recall (trọng số) | Recall | FA/h nền | Trễ median / p90 (s) | Trễ OK |
|---|---|---|---|---|---|---|
| smooth=1 k/n=2/2 margin=– | 10 | 0.519 | 0.477 | 9.5 | 1.39 / 5.06 | ✓ |
| smooth=1 k/n=2/3 margin=– | 10 | 0.535 | 0.492 | 10.0 | 1.37 / 5.06 | ✓ |
| smooth=1 k/n=2/2 margin=– | 15 | 0.535 | 0.500 | 14.0 | 1.27 / 5.06 | ✓ |
| smooth=1 k/n=2/3 margin=– | 15 | 0.551 | 0.515 | 14.5 | 1.27 / 5.06 | ✓ |
| smooth=1 k/n=1/1 margin=– | 15 | 0.556 | 0.531 | 15.0 | 1.17 / 4.21 | ✓ |
| smooth=1 k/n=2/3 margin=– | 22 | 0.578 | 0.554 | 21.0 | 1.27 / 4.53 | ✓ |
| smooth=1 k/n=2/2 margin=– | 22 | 0.572 | 0.546 | 21.5 | 1.27 / 4.60 | ✓ |
| smooth=1 k/n=1/1 margin=– | 22 | 0.583 | 0.554 | 22.0 | 0.99 / 4.09 | ✓ |
## Tất cả điểm (theo ngân sách)
### Ngân sách ≤ 10 báo nhầm / giờ
| Hậu xử lý | Khả thi | Recall (trọng số) | Recall | FA/h nền | Trễ median (s) |
|---|---|---|---|---|---|
| smooth=1 k/n=2/3 margin=– | ✓ | 0.535 | 0.492 | 10.0 | 1.37 |
| smooth=1 k/n=2/2 margin=– | ✓ | 0.519 | 0.477 | 9.5 | 1.39 |
| smooth=2 k/n=1/1 margin=– | ✓ | 0.519 | 0.485 | 10.0 | 1.48 |
| smooth=3 k/n=1/1 margin=– | ✓ | 0.497 | 0.454 | 10.0 | 1.77 ✗ |
| smooth=1 k/n=3/4 margin=– | ✓ | 0.481 | 0.423 | 10.0 | 1.58 ✗ |
| smooth=3 k/n=2/2 margin=– | ✓ | 0.481 | 0.431 | 9.5 | 1.94 ✗ |
| smooth=3 k/n=2/3 margin=– | ✓ | 0.481 | 0.431 | 9.5 | 1.94 ✗ |
| smooth=2 k/n=2/2 margin=– | ✓ | 0.476 | 0.423 | 9.5 | 1.67 ✗ |
| smooth=2 k/n=2/3 margin=– | ✓ | 0.476 | 0.423 | 9.5 | 1.67 ✗ |
| smooth=2 k/n=3/4 margin=– | ✓ | 0.471 | 0.408 | 9.5 | 1.87 ✗ |
| smooth=3 k/n=3/4 margin=– | ✓ | 0.455 | 0.385 | 9.5 | 2.29 ✗ |
| smooth=1 k/n=1/1 margin=– | ✗ | 0.508 | 0.477 | 12.0 | 1.25 |
### Ngân sách ≤ 15 báo nhầm / giờ
| Hậu xử lý | Khả thi | Recall (trọng số) | Recall | FA/h nền | Trễ median (s) |
|---|---|---|---|---|---|
| smooth=1 k/n=1/1 margin=– | ✓ | 0.556 | 0.531 | 15.0 | 1.17 |
| smooth=1 k/n=2/3 margin=– | ✓ | 0.551 | 0.515 | 14.5 | 1.27 |
| smooth=3 k/n=1/1 margin=– | ✓ | 0.551 | 0.531 | 15.0 | 1.79 ✗ |
| smooth=2 k/n=1/1 margin=– | ✓ | 0.545 | 0.508 | 14.5 | 1.37 |
| smooth=1 k/n=2/2 margin=– | ✓ | 0.535 | 0.500 | 14.0 | 1.27 |
| smooth=3 k/n=2/2 margin=– | ✓ | 0.519 | 0.477 | 15.0 | 1.98 ✗ |
| smooth=3 k/n=2/3 margin=– | ✓ | 0.519 | 0.477 | 15.0 | 1.98 ✗ |
| smooth=3 k/n=3/4 margin=– | ✓ | 0.519 | 0.477 | 14.5 | 2.29 ✗ |
| smooth=1 k/n=3/4 margin=– | ✓ | 0.513 | 0.462 | 15.0 | 1.73 ✗ |
| smooth=2 k/n=3/4 margin=– | ✓ | 0.508 | 0.462 | 15.0 | 1.97 ✗ |
| smooth=2 k/n=2/2 margin=– | ✓ | 0.503 | 0.454 | 14.5 | 1.56 ✗ |
| smooth=2 k/n=2/3 margin=– | ✓ | 0.503 | 0.454 | 14.5 | 1.56 ✗ |
### Ngân sách ≤ 22 báo nhầm / giờ
| Hậu xử lý | Khả thi | Recall (trọng số) | Recall | FA/h nền | Trễ median (s) |
|---|---|---|---|---|---|
| smooth=1 k/n=1/1 margin=– | ✓ | 0.583 | 0.554 | 22.0 | 0.99 |
| smooth=1 k/n=2/3 margin=– | ✓ | 0.578 | 0.554 | 21.0 | 1.27 |
| smooth=2 k/n=1/1 margin=– | ✓ | 0.578 | 0.554 | 22.0 | 1.32 |
| smooth=1 k/n=2/2 margin=– | ✓ | 0.572 | 0.546 | 21.5 | 1.27 |
| smooth=2 k/n=2/2 margin=– | ✓ | 0.572 | 0.546 | 22.0 | 1.71 ✗ |
| smooth=2 k/n=2/3 margin=– | ✓ | 0.572 | 0.546 | 22.0 | 1.71 ✗ |
| smooth=3 k/n=1/1 margin=– | ✓ | 0.572 | 0.554 | 21.5 | 1.78 ✗ |
| smooth=3 k/n=2/2 margin=– | ✓ | 0.561 | 0.531 | 21.0 | 1.98 ✗ |
| smooth=3 k/n=2/3 margin=– | ✓ | 0.561 | 0.538 | 21.5 | 1.88 ✗ |
| smooth=1 k/n=3/4 margin=– | ✓ | 0.551 | 0.515 | 21.0 | 1.77 ✗ |
| smooth=2 k/n=3/4 margin=– | ✓ | 0.545 | 0.515 | 22.0 | 1.89 ✗ |
| smooth=3 k/n=3/4 margin=– | ✓ | 0.535 | 0.500 | 20.5 | 2.27 ✗ |
