# Tinh chỉnh hậu xử lý (WP8) – kịch bản **val**

Model `artifacts/models/student_int8.tflite`; kịch bản `artifacts/data/scenarios_val`: 30 kịch bản có sự kiện (130 sự kiện), 20 kịch bản chỉ có nền (2.00 giờ). Mục tiêu: tối đa recall có trọng số (DANGER ×2) với báo nhầm nền ≤ ngân sách và trễ median ≤ 1.5 s. Ngưỡng từng lớp chọn tham lam trên lưới 0.3–0.99. Sinh bởi `python -m soundwatch tune`.

## Cấu hình tốt nhất theo ngân sách báo nhầm

| Ngân sách FA/h | Hậu xử lý | Recall (trọng số) | Recall | Recall DANGER TB / min | FA/h nền | Trễ median / p90 (s) | Ngưỡng |
|---|---|---|---|---|---|---|---|
| ≤ 2 | (không có điểm khả thi) | | | | | | |
| ≤ 5 | (không có điểm khả thi) | | | | | | |
| ≤ 10 | (không có điểm khả thi) | | | | | | |
## Pareto front (recall có trọng số ↑, FA/h ↓, trễ median ↓)
Chỉ các điểm khả thi với ngân sách của chúng; ✓ = trễ median ≤ 1.5 s.
| Hậu xử lý | Ngân sách | Recall (trọng số) | Recall | FA/h nền | Trễ median / p90 (s) | Trễ OK |
|---|---|---|---|---|---|---|
| smooth=3 k/n=2/2 margin=– | 5 | 0.433 | 0.377 | 5.0 | 2.08 / 5.01 | ✗ |
| smooth=3 k/n=2/2 margin=0.2 | 5 | 0.433 | 0.377 | 5.0 | 2.08 / 5.01 | ✗ |
| smooth=3 k/n=2/2 margin=0.4 | 5 | 0.433 | 0.377 | 5.0 | 2.08 / 5.01 | ✗ |
| smooth=3 k/n=2/3 margin=– | 5 | 0.433 | 0.377 | 5.0 | 2.08 / 5.01 | ✗ |
| smooth=3 k/n=2/3 margin=0.2 | 5 | 0.433 | 0.377 | 5.0 | 2.08 / 5.01 | ✗ |
| smooth=3 k/n=2/3 margin=0.4 | 5 | 0.433 | 0.377 | 5.0 | 2.08 / 5.01 | ✗ |
| smooth=2 k/n=3/4 margin=– | 5 | 0.390 | 0.338 | 5.0 | 1.97 / 5.17 | ✗ |
| smooth=2 k/n=3/4 margin=0.2 | 5 | 0.390 | 0.338 | 5.0 | 1.97 / 5.17 | ✗ |
| smooth=2 k/n=3/4 margin=0.4 | 5 | 0.390 | 0.338 | 5.0 | 1.97 / 5.17 | ✗ |
| smooth=3 k/n=2/3 margin=0.4 | 10 | 0.471 | 0.408 | 9.0 | 2.08 / 4.85 | ✗ |
| smooth=3 k/n=2/2 margin=– | 10 | 0.508 | 0.462 | 9.5 | 1.99 / 4.23 | ✗ |
| smooth=3 k/n=2/2 margin=0.2 | 10 | 0.508 | 0.462 | 9.5 | 1.99 / 4.23 | ✗ |
| smooth=3 k/n=2/3 margin=– | 10 | 0.508 | 0.462 | 9.5 | 1.99 / 4.23 | ✗ |
| smooth=3 k/n=2/3 margin=0.2 | 10 | 0.508 | 0.462 | 9.5 | 1.99 / 4.23 | ✗ |
| smooth=2 k/n=3/4 margin=– | 10 | 0.503 | 0.454 | 9.5 | 1.87 / 4.23 | ✗ |
| smooth=2 k/n=2/2 margin=– | 10 | 0.444 | 0.408 | 9.5 | 1.56 / 5.17 | ✗ |
| smooth=2 k/n=2/2 margin=0.2 | 10 | 0.444 | 0.408 | 9.5 | 1.56 / 5.17 | ✗ |
| smooth=2 k/n=2/2 margin=0.4 | 10 | 0.444 | 0.408 | 9.5 | 1.56 / 5.17 | ✗ |
| smooth=2 k/n=2/3 margin=– | 10 | 0.444 | 0.408 | 9.5 | 1.56 / 5.17 | ✗ |
| smooth=2 k/n=2/3 margin=0.2 | 10 | 0.444 | 0.408 | 9.5 | 1.56 / 5.17 | ✗ |
| smooth=2 k/n=2/3 margin=0.4 | 10 | 0.444 | 0.408 | 9.5 | 1.56 / 5.17 | ✗ |
| smooth=3 k/n=1/1 margin=– | 10 | 0.465 | 0.423 | 10.0 | 1.77 / 4.16 | ✗ |
| smooth=3 k/n=1/1 margin=0.2 | 10 | 0.465 | 0.423 | 10.0 | 1.77 / 4.16 | ✗ |
| smooth=3 k/n=1/1 margin=0.4 | 10 | 0.465 | 0.423 | 10.0 | 1.77 / 4.16 | ✗ |
## Tất cả điểm (theo ngân sách)
### Ngân sách ≤ 2 báo nhầm / giờ
| Hậu xử lý | Khả thi | Recall (trọng số) | Recall | FA/h nền | Trễ median (s) |
|---|---|---|---|---|---|
| smooth=1 k/n=1/1 margin=– | ✗ | 0.578 | 0.569 | 46.0 | 0.97 |
| smooth=1 k/n=1/1 margin=0.2 | ✗ | 0.578 | 0.569 | 46.0 | 0.97 |
| smooth=1 k/n=1/1 margin=0.4 | ✗ | 0.578 | 0.569 | 46.0 | 0.97 |
| smooth=1 k/n=2/3 margin=– | ✗ | 0.535 | 0.515 | 20.0 | 1.23 |
| smooth=1 k/n=2/3 margin=0.2 | ✗ | 0.535 | 0.515 | 20.0 | 1.23 |
| smooth=1 k/n=2/3 margin=0.4 | ✗ | 0.535 | 0.515 | 20.0 | 1.23 |
| smooth=1 k/n=2/2 margin=– | ✗ | 0.524 | 0.500 | 19.5 | 1.23 |
| smooth=1 k/n=2/2 margin=0.2 | ✗ | 0.524 | 0.500 | 19.5 | 1.23 |
| smooth=1 k/n=2/2 margin=0.4 | ✗ | 0.524 | 0.500 | 19.5 | 1.23 |
| smooth=2 k/n=1/1 margin=– | ✗ | 0.503 | 0.492 | 18.0 | 1.43 |
| smooth=2 k/n=1/1 margin=0.2 | ✗ | 0.503 | 0.492 | 18.0 | 1.43 |
| smooth=2 k/n=1/1 margin=0.4 | ✗ | 0.503 | 0.492 | 18.0 | 1.43 |
| smooth=1 k/n=3/4 margin=– | ✗ | 0.481 | 0.438 | 13.5 | 1.48 |
| smooth=1 k/n=3/4 margin=0.2 | ✗ | 0.481 | 0.438 | 13.5 | 1.48 |
| smooth=1 k/n=3/4 margin=0.4 | ✗ | 0.481 | 0.438 | 13.5 | 1.48 |
| smooth=2 k/n=2/2 margin=– | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/2 margin=0.2 | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/2 margin=0.4 | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/3 margin=– | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/3 margin=0.2 | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/3 margin=0.4 | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=3/4 margin=– | ✗ | 0.369 | 0.323 | 5.0 | 1.91 ✗ |
| smooth=2 k/n=3/4 margin=0.2 | ✗ | 0.369 | 0.323 | 5.0 | 1.91 ✗ |
| smooth=2 k/n=3/4 margin=0.4 | ✗ | 0.369 | 0.323 | 5.0 | 1.91 ✗ |
| smooth=3 k/n=1/1 margin=– | ✗ | 0.369 | 0.354 | 8.0 | 1.84 ✗ |
| smooth=3 k/n=1/1 margin=0.2 | ✗ | 0.369 | 0.354 | 8.0 | 1.84 ✗ |
| smooth=3 k/n=1/1 margin=0.4 | ✗ | 0.369 | 0.354 | 8.0 | 1.84 ✗ |
| smooth=3 k/n=2/2 margin=– | ✗ | 0.316 | 0.285 | 3.0 | 2.03 ✗ |
| smooth=3 k/n=2/2 margin=0.2 | ✗ | 0.316 | 0.285 | 3.0 | 2.03 ✗ |
| smooth=3 k/n=2/2 margin=0.4 | ✗ | 0.316 | 0.285 | 3.0 | 2.03 ✗ |
| smooth=3 k/n=2/3 margin=– | ✗ | 0.316 | 0.285 | 3.0 | 2.03 ✗ |
| smooth=3 k/n=2/3 margin=0.2 | ✗ | 0.316 | 0.285 | 3.0 | 2.03 ✗ |
| smooth=3 k/n=2/3 margin=0.4 | ✗ | 0.316 | 0.285 | 3.0 | 2.03 ✗ |
| smooth=3 k/n=3/4 margin=– | ✗ | 0.267 | 0.223 | 3.0 | 2.29 ✗ |
| smooth=3 k/n=3/4 margin=0.2 | ✗ | 0.267 | 0.223 | 3.0 | 2.29 ✗ |
| smooth=3 k/n=3/4 margin=0.4 | ✗ | 0.267 | 0.223 | 3.0 | 2.29 ✗ |
### Ngân sách ≤ 5 báo nhầm / giờ
| Hậu xử lý | Khả thi | Recall (trọng số) | Recall | FA/h nền | Trễ median (s) |
|---|---|---|---|---|---|
| smooth=3 k/n=2/2 margin=– | ✓ | 0.433 | 0.377 | 5.0 | 2.08 ✗ |
| smooth=3 k/n=2/2 margin=0.2 | ✓ | 0.433 | 0.377 | 5.0 | 2.08 ✗ |
| smooth=3 k/n=2/2 margin=0.4 | ✓ | 0.433 | 0.377 | 5.0 | 2.08 ✗ |
| smooth=3 k/n=2/3 margin=– | ✓ | 0.433 | 0.377 | 5.0 | 2.08 ✗ |
| smooth=3 k/n=2/3 margin=0.2 | ✓ | 0.433 | 0.377 | 5.0 | 2.08 ✗ |
| smooth=3 k/n=2/3 margin=0.4 | ✓ | 0.433 | 0.377 | 5.0 | 2.08 ✗ |
| smooth=3 k/n=3/4 margin=– | ✓ | 0.396 | 0.331 | 5.0 | 2.48 ✗ |
| smooth=3 k/n=3/4 margin=0.2 | ✓ | 0.396 | 0.331 | 5.0 | 2.48 ✗ |
| smooth=2 k/n=3/4 margin=– | ✓ | 0.390 | 0.338 | 5.0 | 1.97 ✗ |
| smooth=2 k/n=3/4 margin=0.2 | ✓ | 0.390 | 0.338 | 5.0 | 1.97 ✗ |
| smooth=2 k/n=3/4 margin=0.4 | ✓ | 0.390 | 0.338 | 5.0 | 1.97 ✗ |
| smooth=3 k/n=3/4 margin=0.4 | ✓ | 0.364 | 0.292 | 5.0 | 2.48 ✗ |
| smooth=1 k/n=1/1 margin=– | ✗ | 0.578 | 0.569 | 46.0 | 0.97 |
| smooth=1 k/n=1/1 margin=0.2 | ✗ | 0.578 | 0.569 | 46.0 | 0.97 |
| smooth=1 k/n=1/1 margin=0.4 | ✗ | 0.578 | 0.569 | 46.0 | 0.97 |
| smooth=1 k/n=2/3 margin=– | ✗ | 0.535 | 0.515 | 20.0 | 1.23 |
| smooth=1 k/n=2/3 margin=0.2 | ✗ | 0.535 | 0.515 | 20.0 | 1.23 |
| smooth=1 k/n=2/3 margin=0.4 | ✗ | 0.535 | 0.515 | 20.0 | 1.23 |
| smooth=1 k/n=2/2 margin=– | ✗ | 0.524 | 0.500 | 19.5 | 1.23 |
| smooth=1 k/n=2/2 margin=0.2 | ✗ | 0.524 | 0.500 | 19.5 | 1.23 |
| smooth=1 k/n=2/2 margin=0.4 | ✗ | 0.524 | 0.500 | 19.5 | 1.23 |
| smooth=2 k/n=1/1 margin=– | ✗ | 0.503 | 0.492 | 18.0 | 1.43 |
| smooth=2 k/n=1/1 margin=0.2 | ✗ | 0.503 | 0.492 | 18.0 | 1.43 |
| smooth=2 k/n=1/1 margin=0.4 | ✗ | 0.503 | 0.492 | 18.0 | 1.43 |
| smooth=1 k/n=3/4 margin=– | ✗ | 0.481 | 0.438 | 13.5 | 1.48 |
| smooth=1 k/n=3/4 margin=0.2 | ✗ | 0.481 | 0.438 | 13.5 | 1.48 |
| smooth=1 k/n=3/4 margin=0.4 | ✗ | 0.481 | 0.438 | 13.5 | 1.48 |
| smooth=2 k/n=2/2 margin=– | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/2 margin=0.2 | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/2 margin=0.4 | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/3 margin=– | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/3 margin=0.2 | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=2 k/n=2/3 margin=0.4 | ✗ | 0.422 | 0.392 | 9.5 | 1.55 ✗ |
| smooth=3 k/n=1/1 margin=– | ✗ | 0.369 | 0.354 | 8.0 | 1.84 ✗ |
| smooth=3 k/n=1/1 margin=0.2 | ✗ | 0.369 | 0.354 | 8.0 | 1.84 ✗ |
| smooth=3 k/n=1/1 margin=0.4 | ✗ | 0.369 | 0.354 | 8.0 | 1.84 ✗ |
### Ngân sách ≤ 10 báo nhầm / giờ
| Hậu xử lý | Khả thi | Recall (trọng số) | Recall | FA/h nền | Trễ median (s) |
|---|---|---|---|---|---|
| smooth=3 k/n=2/2 margin=– | ✓ | 0.508 | 0.462 | 9.5 | 1.99 ✗ |
| smooth=3 k/n=2/2 margin=0.2 | ✓ | 0.508 | 0.462 | 9.5 | 1.99 ✗ |
| smooth=3 k/n=2/3 margin=– | ✓ | 0.508 | 0.462 | 9.5 | 1.99 ✗ |
| smooth=3 k/n=2/3 margin=0.2 | ✓ | 0.508 | 0.462 | 9.5 | 1.99 ✗ |
| smooth=2 k/n=3/4 margin=– | ✓ | 0.503 | 0.454 | 9.5 | 1.87 ✗ |
| smooth=3 k/n=3/4 margin=– | ✓ | 0.481 | 0.431 | 10.0 | 2.24 ✗ |
| smooth=3 k/n=2/2 margin=0.4 | ✓ | 0.471 | 0.415 | 10.0 | 2.20 ✗ |
| smooth=3 k/n=2/3 margin=0.4 | ✓ | 0.471 | 0.408 | 9.0 | 2.08 ✗ |
| smooth=2 k/n=3/4 margin=0.2 | ✓ | 0.465 | 0.400 | 10.0 | 2.02 ✗ |
| smooth=3 k/n=1/1 margin=– | ✓ | 0.465 | 0.423 | 10.0 | 1.77 ✗ |
| smooth=3 k/n=1/1 margin=0.2 | ✓ | 0.465 | 0.423 | 10.0 | 1.77 ✗ |
| smooth=3 k/n=1/1 margin=0.4 | ✓ | 0.465 | 0.423 | 10.0 | 1.77 ✗ |
| smooth=2 k/n=3/4 margin=0.4 | ✓ | 0.460 | 0.392 | 9.5 | 1.98 ✗ |
| smooth=3 k/n=3/4 margin=0.2 | ✓ | 0.455 | 0.385 | 10.0 | 2.48 ✗ |
| smooth=2 k/n=2/2 margin=– | ✓ | 0.444 | 0.408 | 9.5 | 1.56 ✗ |
| smooth=2 k/n=2/2 margin=0.2 | ✓ | 0.444 | 0.408 | 9.5 | 1.56 ✗ |
| smooth=2 k/n=2/2 margin=0.4 | ✓ | 0.444 | 0.408 | 9.5 | 1.56 ✗ |
| smooth=2 k/n=2/3 margin=– | ✓ | 0.444 | 0.408 | 9.5 | 1.56 ✗ |
| smooth=2 k/n=2/3 margin=0.2 | ✓ | 0.444 | 0.408 | 9.5 | 1.56 ✗ |
| smooth=2 k/n=2/3 margin=0.4 | ✓ | 0.444 | 0.408 | 9.5 | 1.56 ✗ |
| smooth=3 k/n=3/4 margin=0.4 | ✓ | 0.412 | 0.346 | 10.0 | 2.48 ✗ |
| smooth=1 k/n=1/1 margin=– | ✗ | 0.578 | 0.569 | 46.0 | 0.97 |
| smooth=1 k/n=1/1 margin=0.2 | ✗ | 0.578 | 0.569 | 46.0 | 0.97 |
| smooth=1 k/n=1/1 margin=0.4 | ✗ | 0.578 | 0.569 | 46.0 | 0.97 |
| smooth=1 k/n=2/3 margin=– | ✗ | 0.535 | 0.515 | 20.0 | 1.23 |
| smooth=1 k/n=2/3 margin=0.2 | ✗ | 0.535 | 0.515 | 20.0 | 1.23 |
| smooth=1 k/n=2/3 margin=0.4 | ✗ | 0.535 | 0.515 | 20.0 | 1.23 |
| smooth=1 k/n=2/2 margin=– | ✗ | 0.524 | 0.500 | 19.5 | 1.23 |
| smooth=1 k/n=2/2 margin=0.2 | ✗ | 0.524 | 0.500 | 19.5 | 1.23 |
| smooth=1 k/n=2/2 margin=0.4 | ✗ | 0.524 | 0.500 | 19.5 | 1.23 |
| smooth=2 k/n=1/1 margin=– | ✗ | 0.503 | 0.492 | 18.0 | 1.43 |
| smooth=2 k/n=1/1 margin=0.2 | ✗ | 0.503 | 0.492 | 18.0 | 1.43 |
| smooth=2 k/n=1/1 margin=0.4 | ✗ | 0.503 | 0.492 | 18.0 | 1.43 |
| smooth=1 k/n=3/4 margin=– | ✗ | 0.481 | 0.438 | 13.5 | 1.48 |
| smooth=1 k/n=3/4 margin=0.2 | ✗ | 0.481 | 0.438 | 13.5 | 1.48 |
| smooth=1 k/n=3/4 margin=0.4 | ✗ | 0.481 | 0.438 | 13.5 | 1.48 |
