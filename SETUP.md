# Setup autoresearch (bản chạy trên CPU)

Repo này port [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (commit `228791f`, license MIT) sang chạy **chỉ bằng CPU**, theo các gợi ý "smaller compute" của Karpathy trong `README.md`.

## Khác gì so với bản gốc

| | Bản gốc (GPU) | Bản này (CPU) |
|---|---|---|
| Dữ liệu | climbmix (HuggingFace) | TinyStories (HuggingFace), hoặc TinyShakespeare (GitHub, 1MB) |
| Tokenizer | BPE 8192 | BPE 2048 |
| `MAX_SEQ_LEN` | 2048 | 256 |
| Model | `DEPTH` 8, ~50M tham số | `DEPTH` 4, ~5M tham số |
| Attention | Flash Attention 3, `SSSL` | PyTorch SDPA, `L` |
| Batch | 2**19 token/step | 2**14 token/step |
| Kiểu số | bfloat16 | float32 |
| Output | `peak_vram_mb`, `mfu_percent` | `peak_mem_mb` (RAM), `tok_per_sec` |

Giữ nguyên: ngân sách 5 phút mỗi lần train, metric `val_bpb`, optimizer Muon + AdamW, và quy trình agent trong `program.md`.

## Yêu cầu

- Python 3.10+, [uv](https://docs.astral.sh/uv/), một trình biên dịch C/C++ (gcc/clang) cho `torch.compile`. Nếu không có, đặt `COMPILE = False` trong `train.py`.
- Khoảng 2GB RAM trống. Càng nhiều nhân CPU càng nhanh.

## Chạy

```bash
# 1. Cài uv (nếu chưa có)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Cài dependencies
uv sync

# 3. Chuẩn bị dữ liệu (một lần; lưu ở ~/.cache/autoresearch-cpu/)
uv run prepare.py                        # TinyStories, 200MB đầu của tập train
uv run prepare.py --dataset shakespeare  # hoặc: TinyShakespeare (nhỏ, tải từ GitHub)

# 4. Chạy thử một lần training (~5 phút train + ~1 phút compile/eval)
uv run train.py
```

Chạy lại `prepare.py` với dataset khác sẽ ghi đè dữ liệu cũ. Kết quả `val_bpb` chỉ so sánh được trong cùng một dataset.

## Bật chế độ tự nghiên cứu

Mở Claude Code (hoặc agent khác) trong repo này, rồi gửi prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

Agent sẽ tạo branch `autoresearch/<tag>`, sửa `train.py`, mỗi lần train 5 phút, giữ lại thay đổi nếu `val_bpb` giảm, rồi ghi kết quả vào `results.tsv`.

- Bạn chỉnh `program.md` (hướng dẫn cho agent). Agent chỉ được sửa `train.py`.
- `prepare.py` (dữ liệu, tokenizer, hàm đánh giá) là cố định, không sửa.

## Baseline đã đo

Máy cloud 4 nhân CPU, 15GB RAM, dataset `shakespeare`: `val_bpb` 2.611, 187 step, khoảng 9.6K token/giây, RAM đỉnh 1.7GB. Train loss xuống thấp hơn nhiều so với val, vì trong 5 phút model đã xem 1MB dữ liệu khoảng 10 lần (overfit). Đây chính là kiểu vấn đề để agent tự tìm cách cải thiện. Với TinyStories, dữ liệu lớn hơn nhiều nên không bị vậy.
