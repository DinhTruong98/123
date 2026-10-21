# Setup autoresearch

Repo này chứa [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (commit `228791f`, license MIT), giữ nguyên bản gốc. Đọc `README.md` để biết chi tiết.

## Yêu cầu

- 1 GPU NVIDIA (bản gốc test trên H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).
- Không chạy được trên máy không có GPU (ví dụ máy cloud của Claude Code on the web). Với Mac / Windows / AMD, xem mục "Notable forks" trong `README.md`.

## Chạy

```bash
# 1. Cài uv (nếu chưa có)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Cài dependencies
uv sync

# 3. Tải dữ liệu + train tokenizer (một lần, ~2 phút; lưu ở ~/.cache/autoresearch/)
uv run prepare.py

# 4. Chạy thử một lần training (~5 phút) để kiểm tra setup
uv run train.py
```

## Bật chế độ tự nghiên cứu

Mở Claude Code (hoặc agent khác) trong repo này trên máy có GPU, rồi gửi prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

Agent sẽ tạo branch `autoresearch/<tag>`, sửa `train.py`, mỗi lần train 5 phút, giữ lại thay đổi nếu `val_bpb` giảm, rồi ghi kết quả vào `results.tsv`.

- Bạn chỉnh `program.md` (hướng dẫn cho agent). Agent chỉ được sửa `train.py`.
- `prepare.py` (dữ liệu, tokenizer, hàm đánh giá) là cố định, không sửa.
