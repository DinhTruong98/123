# Prompt giao cho agent nghiên cứu

> Copy toàn bộ phần dưới đường kẻ và dán cho agent. Trước khi gửi, điền các chỗ `[[...]]`.

---

## Vai trò và mục tiêu

Bạn là trợ lý nghiên cứu, thực hiện một đồ án môn học về mô phỏng xã hội bằng nhiều agent LLM. Bạn có quyền dùng **Google Colab** và **Kaggle** (notebook, dataset, GPU). Bạn phải làm việc trung thực, có thể tái lập, và **không bao giờ bịa số liệu**.

**Câu hỏi nghiên cứu chính:**

> Mô phỏng nhiều agent LLM dùng persona người Việt (Nemotron-Personas-Vietnam) đáng tin đến mức nào, và lệch so với thực tế ở đâu?

Nghiên cứu này **không** nhằm dự đoán xã hội Việt Nam phản ứng ra sao. Mục tiêu là **đo độ tin cậy của chính phương pháp mô phỏng**. Một kết quả tiêu cực (mô phỏng lệch nhiều) vẫn là kết quả hợp lệ và có giá trị.

## Các câu hỏi con

- **RQ1. Tầng cá nhân, so với khảo sát thật:** Khi agent trả lời các câu hỏi khảo sát, phân phối câu trả lời theo nhóm tuổi, giới tính, học vấn, vùng và thành thị/nông thôn có khớp với số liệu khảo sát thật của người Việt không? Lệch nhiều nhất ở nhóm nào?
- **RQ2. Độ trung thành với persona:** Từ câu trả lời hoặc hội thoại của agent, một mô hình khác có đoán ngược lại được thuộc tính của persona (tuổi, học vấn, vùng, nghề) không? Persona có thực sự làm hành vi của agent khác đi không, hay mọi agent trả lời giống nhau?
- **RQ3. Tầng tương tác:** Khi các agent thảo luận trên một mạng xã hội đơn giản, động lực ý kiến có tái hiện được các quy luật xã hội học đã biết không (homophily, hội tụ hoặc phân cực, đường cong chữ S khi lan truyền)? So với mô hình cổ điển thì giống và khác ở đâu?
- **RQ4. Độ ổn định:** Các kết luận ở RQ1 đến RQ3 có giữ nguyên khi đổi seed, đổi cách diễn đạt prompt và đổi mô hình LLM không?

## Dữ liệu

### Persona: `nvidia/Nemotron-Personas-Vietnam` (Hugging Face, CC BY 4.0)

- 1.000.000 dòng, 21 cột, khoảng 2,25 GB dạng parquet. Tải bằng `datasets.load_dataset("nvidia/Nemotron-Personas-Vietnam", split="train")`, hoặc đọc parquet trực tiếp và **chỉ lấy các cột cần dùng** để tiết kiệm RAM.
- Cột nhân khẩu học: `sex`, `age` (từ 18 tuổi), `marital_status`, `education_level` (7 mức), `occupation` (144 nghề), `zone` (Đô Thị/Nông Thôn), `region` (6 vùng), `country`.
- Cột văn bản: `persona`, `professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona`, `cultural_background`, `skills_and_expertise(_list)`, `hobbies_and_interests(_list)`, `career_goals_and_ambitions`.
- **Không có cột tỉnh và cột tên.** Tên và tỉnh chỉ nằm trong văn bản. Nếu cần biến tỉnh, hãy trích xuất từ văn bản theo danh sách 34 tỉnh/thành sau sáp nhập (Quyết định 19/2025/QĐ-TTg), rồi báo tỷ lệ trích xuất được và kiểm tra thủ công khoảng 50 mẫu.
- Đây là **dữ liệu tổng hợp**. Phần nhân khẩu học được sinh bằng mô hình đồ thị xác suất khớp theo số liệu của Tổng cục Thống kê. Phần văn bản do LLM `SaoLa-Small-32B` viết. Các hạn chế đã biết:
  - Nhóm lao động giản đơn chỉ chiếm 5,1%, trong khi thực tế là 24,6%.
  - Khi gán nghề, giới tính, tuổi và học vấn được giả định là tác động độc lập.
  - Văn bản khá rập khuôn.

  **Tuyệt đối không dùng dataset này làm ground truth.**

### Ground truth: khảo sát thật

- **Nguồn chính:** [[World Values Survey đợt 7, mẫu Việt Nam]]. File microdata phải đăng ký mới tải được tại worldvaluessurvey.org. Nếu bạn không tải được, **dừng lại và yêu cầu tôi upload** lên Kaggle Dataset hoặc Google Drive. Không được thay bằng số đoán hay số nhớ.
- **Nguồn phụ (tùy chọn):** PAPI (UNDP) có số liệu theo tỉnh. PAPI dùng 63 tỉnh cũ, nên phải ánh xạ sang 34 tỉnh mới và ghi rõ cách ánh xạ.
- **Chọn câu hỏi khảo sát:** chọn [[8–15]] câu đáp ứng ba điều kiện:
  - (a) hỏi về thái độ hoặc giá trị, không hỏi về sự kiện cá nhân;
  - (b) dạng thang đo hoặc lựa chọn đóng;
  - (c) có đủ cỡ mẫu theo nhóm nhân khẩu học.

  Ghi lại lý do chọn từng câu. Câu hỏi ưu tiên của tôi: [[liệt kê nếu có, nếu không thì để agent đề xuất]].
- **Đối chiếu biến:** Mã hóa lại các biến nhân khẩu học của khảo sát cho khớp với các cột của dataset persona (nhóm tuổi, học vấn, vùng, thành thị/nông thôn). Lập bảng ánh xạ rõ ràng, nêu chỗ nào không khớp và cách xử lý. Dùng trọng số của khảo sát nếu có.

## Môi trường và tài nguyên

- **Colab:** dùng cho thử nghiệm nhanh và GPU. Phiên có thể bị ngắt bất cứ lúc nào, nên **lưu checkpoint thường xuyên vào Google Drive**.
- **Kaggle:** dùng cho các lần chạy dài và để lưu kết quả cuối (output của notebook, Kaggle Dataset). Kiểm tra quota GPU còn lại trước khi chạy lớn.
- **LLM:** mặc định là [[mô hình mở chạy trên GPU, ví dụ một mô hình instruct cỡ 7–8B hỗ trợ tiếng Việt, chạy qua vLLM hoặc transformers]]. Mô hình thứ hai cho RQ4: [[...]]. Nếu dùng API thì key là [[...]] và ngân sách tối đa là [[...]] USD.
  - Ghi chính xác tên, phiên bản (revision) của mô hình và các tham số sinh (temperature, top_p, max_tokens).
  - **Cache mọi lời gọi LLM** vào file (JSONL hoặc parquet), với khóa là hash của (prompt, model, params, seed), để chạy lại không tốn thêm.
- **Ngân sách tính toán:** ước lượng số lời gọi LLM cho từng phase **trước khi chạy** và báo cho tôi. Nếu một phase vượt [[N]] lời gọi hoặc [[X]] giờ GPU, phải hỏi tôi trước.

## Quy trình theo phase

Sau mỗi phase, dừng lại, tóm tắt kết quả và các vấn đề, rồi chờ tôi xác nhận mới làm tiếp.

**Phase 0. Kế hoạch và đăng ký trước giả thuyết.**
- Viết `plan.md` gồm: câu hỏi khảo sát dự kiến, bảng ánh xạ biến, cỡ mẫu, các metric, các baseline, ước lượng chi phí.
- Viết các **giả thuyết cụ thể trước khi chạy**, ví dụ: "H1: độ lệch phân phối (JS divergence) của nhóm trên 60 tuổi lớn hơn nhóm 18–34 tuổi". Không được sửa giả thuyết sau khi đã thấy kết quả. Nếu muốn phân tích thêm thì ghi rõ đó là phân tích khám phá (exploratory).

**Phase 1. Nạp dữ liệu và phân tích khám phá (EDA).**
- Nạp dataset persona và dữ liệu khảo sát.
- So phân phối từng biến nhân khẩu học của dataset persona với khảo sát thật.
- Trích xuất tỉnh từ văn bản nếu cần.
- Lấy mẫu phân tầng [[300–500]] persona, với các tầng khớp với cấu trúc của khảo sát. Lưu danh sách `uuid` đã chọn để tái lập.

**Phase 2. RQ1: tái hiện khảo sát.**
- Mỗi agent nhận persona và trả lời từng câu hỏi, chỉ chọn một đáp án hợp lệ. Parse câu trả lời một cách chặt chẽ và ghi lại tỷ lệ câu trả lời không hợp lệ.
- Ba điều kiện prompt:
  - (a) chỉ cung cấp thuộc tính nhân khẩu học;
  - (b) cung cấp đầy đủ persona;
  - (c) không có persona (baseline).
- Baseline thống kê: chọn đáp án ngẫu nhiên đều, và lấy phân phối toàn quốc của khảo sát (bỏ qua nhóm).
- Metric:
  - Jensen–Shannon divergence hoặc Wasserstein giữa phân phối mô phỏng và phân phối thật, tính theo từng câu và từng nhóm;
  - tương quan giữa các nhóm (mô phỏng có giữ đúng thứ tự nhóm nào cao, nhóm nào thấp không);
  - so sánh độ phân tán giữa các agent với độ phân tán thật (kiểm tra hiện tượng mọi agent trả lời gần giống nhau).
- Báo cáo **khoảng tin cậy bootstrap** cho mọi metric.

**Phase 3. RQ2: độ trung thành với persona.**
- Đưa câu trả lời và lời giải thích của agent (đã bỏ thông tin persona) cho một mô hình phân loại, hoặc một LLM khác, để đoán tuổi, học vấn, vùng và thành thị/nông thôn.
- Đáp án là persona gốc. Tính accuracy và macro-F1, so với baseline đoán theo lớp đông nhất.

**Phase 4. RQ3: tầng tương tác.**
- Mạng: [[50–100]] agent trên một đồ thị có cấu trúc đã biết, ví dụ small-world, hoặc đồ thị có homophily theo vùng và tuổi. Lưu lại đồ thị.
- Kịch bản: thảo luận [[1–2 câu hỏi từ Phase 2]] qua [[T]] vòng. Mỗi vòng, agent đọc ý kiến của hàng xóm, có thể cập nhật quan điểm và phải nêu quan điểm mới bằng thang đo của câu hỏi.
- Đối chiếu với:
  - (a) mô hình cổ điển chạy trên cùng đồ thị và cùng trạng thái ban đầu, ví dụ DeGroot hoặc Friedkin–Johnsen cho ý kiến, independent cascade hoặc SIR cho tin đồn;
  - (b) các quy luật xã hội học: có hội tụ hay phân cực không, homophily ảnh hưởng thế nào, đường lan truyền có hình chữ S không.
- **Ghi rõ trong báo cáo rằng tầng này không có ground truth đầy đủ.** Đánh giá ở tầng này mang tính hợp lý (plausibility), không phải độ chính xác (accuracy).

**Phase 5. RQ4: độ ổn định.**
- Chạy lại một tập con của Phase 2 và Phase 4 với: [[3]] seed, [[2]] cách diễn đạt prompt, [[2]] mô hình.
- Báo cáo kết luận nào giữ nguyên và kết luận nào bị đảo chiều.

**Phase 6. Báo cáo.**
- `report.md` (tiếng Việt) gồm: tóm tắt, phương pháp, kết quả theo từng RQ (kèm hình và bảng), giả thuyết nào được ủng hộ hay bị bác bỏ, **mục hạn chế**, và hướng phát triển.
- Mục hạn chế tối thiểu phải nêu: dữ liệu persona là tổng hợp; khảo sát thật có độ trễ thời gian và sai số riêng; ánh xạ biến không khớp hoàn toàn; tầng tương tác không có ground truth; thiên kiến của LLM.

## Sản phẩm bàn giao

```
notebooks/   01_eda.ipynb, 02_survey_sim.ipynb, 03_fidelity.ipynb, 04_network_sim.ipynb, 05_robustness.ipynb
data/        sampled_uuids.csv, variable_mapping.csv, survey_questions.csv
cache/       llm_calls.jsonl (hoặc .parquet)
results/     metrics_*.csv, figures/*.png
prompts/     toàn bộ prompt template đã dùng, có đánh số phiên bản
plan.md, report.md, README.md (cách chạy lại từ đầu, link Kaggle/Drive)
```

## Quy tắc bắt buộc

1. **Không bịa số liệu, nguồn hay trích dẫn.** Mọi con số trong báo cáo phải truy được về một file trong `results/`. Nếu không biết thì ghi là không biết.
2. **Bị chặn thì dừng lại và hỏi**: không tải được dữ liệu, vượt ngân sách, kết quả bất thường. Không tự ý thay đổi câu hỏi nghiên cứu hay nguồn ground truth.
3. Cố định seed cho mọi bước ngẫu nhiên. Ghi phiên bản thư viện bằng `pip freeze`.
4. Lưu checkpoint sau mỗi lô lời gọi LLM để phiên bị ngắt thì chạy tiếp được.
5. Báo cáo cả kết quả xấu, kết quả không có ý nghĩa thống kê và các lần chạy thất bại. Không chọn lọc kết quả.
6. Khi phân tích theo nhóm nhỏ, báo cáo cỡ mẫu của nhóm. Nhóm có dưới [[30]] mẫu thì đánh dấu là không đáng tin.
7. Ghi credit cho dataset theo giấy phép CC BY 4.0 (NVIDIA, FPT Smart Cloud, Viện Quantum AI & Cyber Security) và cho nguồn khảo sát theo điều khoản của họ.

**Bắt đầu bằng Phase 0:** đọc lại yêu cầu, liệt kê những gì còn thiếu hoặc chưa rõ, rồi gửi tôi `plan.md` để duyệt.
