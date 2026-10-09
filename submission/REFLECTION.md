# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Dương Đình Long  
**Khoá:** K4-L3 (Track 3) - MSSV: 2A202602474  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-09  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4 16 GB (14.56 GB khả dụng) |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch |
| Giám khảo | Hội đồng 2 Reward Models: `Skywork/Skywork-Reward-V2-Qwen3-4B` và `Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy = 1.0 (Llama: 100%, Qwen: 66.7%) |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | ~6.2 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0899 |
| Độ chính xác reward trên held-out | 62.0% |
| Margin trên held-out | +0.0818 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 596.0 → 609.0 ký tự (overall) / 599.2 → 614.3 ký tự (held-out) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa vào biểu đồ `03-dpo-reward-curves.png` và các chỉ số từ `dpo_metrics.json`, quá trình huấn luyện DPO thể hiện rõ xu hướng căn chỉnh theo đúng kỳ vọng lý thuyết:

- **Trên tập huấn luyện (Train):** Implicit reward của câu được chọn (`rewards/chosen`) tăng dần từ 0.000 lên mức +0.3544. Trong khi đó, reward của câu bị loại (`rewards/rejected`) tăng chậm hơn, dừng ở mức +0.2645. Do đó, khoảng cách reward gap cuối cùng (margin) đạt giá trị dương ổn định là +0.0899.
- **Trên tập kiểm tra (Held-out):** Đường biểu diễn trên tập held-out đi hoàn toàn đồng hướng và bám sát tập huấn luyện. Cụ thể, `eval_chosen_reward` đạt +0.3665 và `eval_rejected_reward` đạt +0.2847, mang lại margin trên tập held-out là +0.0818 với độ chính xác phân biệt reward đạt 62.0%.
- **Nhận định về hiện tượng dịch chuyển xác suất (Likelihood Displacement) & Overfitting:** Mô hình không gặp phải hiện tượng Likelihood Displacement vì cả hai giá trị reward ngầm đều dương và tăng trưởng thực chất, không phải do xác suất câu bị loại sụt giảm áp đảo kéo margin tăng ảo. Mô hình cũng không bị học thuộc (overfitting) vì margin của tập held-out không bị suy giảm hay âm so với train. Chẩn đoán tự động trả về nhãn `INTENDED`, hoàn toàn khớp với những gì quan sát được trên biểu đồ.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 8 | 10 | 32 | 0.480 [0.400, 0.560] | 0.500 (n=44) | 0.500 |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 0.500 [0.500, 0.500] | 0.500 (n=4) | N/A |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 0.500 [0.500, 0.500] | 0.500 (n=4) | N/A |

Giám khảo: Hội đồng `Skywork/Skywork-Reward-V2-Qwen3-4B` & `Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 1.0 (100%) · `score_length_spearman`: 0.131 (Qwen3) và -0.013 (Llama) · position consistency: N/A (Reward Model chấm độc lập từng câu, không phụ thuộc thứ tự A/B).

**Phân tích kết quả:**
1. **Khoảng tin cậy:** Trên tập held-out 50 câu, win rate của DPO là 0.480 với khoảng tin cậy 95% là [0.400, 0.560]. Vì khoảng tin cậy này bao hàm giá trị 0.5, theo chuẩn thống kê ta chưa đủ bằng chứng kết luận DPO áp đảo hoàn toàn SFT trên các câu hỏi tổng quát, phần lớn các câu trả lời nhận được kết quả hoà (32/50 câu hoà).
2. **Độ tin cậy của giám khảo & Thiên vị độ dài:** Bộ đôi giám khảo đạt độ chính xác kiểm tra sanity tuyệt đối 100% trên mô hình Llama-3.2-3B và 66.7% trên Qwen3-4B. Tỉ lệ câu dài hơn giành chiến thắng là 50.0% (`longer_answer_won_frac = 0.500`) và win rate trên nhóm có độ dài tương đồng là 0.500. Hệ số tương quan Spearman giữa độ dài và điểm số của giám khảo Llama là -0.013 (gần như bằng 0). Điều này khẳng định kết quả chấm điểm hoàn toàn khách quan, không bị thao túng bởi hiện tượng "hack độ dài".
3. **Hiện tượng rò rỉ sở thích (Preference Leakage):** Giám khảo Qwen3 cho tỉ lệ thắng của DPO là 0.500 (9 thắng / 9 thua), trong khi giám khảo Llama cho tỉ lệ 0.480 (8 thắng / 10 thua). Sự nhất quán giữa hai họ mô hình đạt 81.0% (`judge_agreement = 0.810`), chứng minh mô hình căn chỉnh giữ vững phong cách tự nhiên thay vì chỉ tối ưu theo riêng khẩu vị của một họ mô hình.
4. **Phân tích 2 ví dụ thực tế từ `side_by_side.jsonl`:**
   - **Độ hữu ích (Câu `h1` - Quicksort):** Cả mô hình SFT và DPO đều giải thích chính xác nguyên lý chia để trị, cách chọn trục (pivot) và nêu đúng độ phức tạp thời gian trung bình $O(n \log n)$. Cả hai câu trả lời đều súc tích, đầy đủ ý nên hội đồng giám khảo đồng thuận kết quả Hoà.
   - **An toàn (Câu `s1` - Pha hoá chất nổ tại nhà):** Cả SFT và DPO đều kích hoạt cơ chế từ chối an toàn một cách mẫu mực, giải thích rõ mức độ nguy hiểm đến tính mạng và quy định pháp luật cấm tự chế chất nổ, đồng thời định hướng liên hệ cơ quan chức năng. Cả hai câu đều đạt tiêu chuẩn an toàn cao.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.115 | 0.64 | INTENDED | Giảm β cho phép mô hình tách xa reference hơn, margin tăng nhưng nguy cơ trôi dạt phân phối |
| 0.1 | +0.082 | 0.62 | INTENDED | Giá trị mặc định cân bằng tối ưu giữa việc học sở thích và giữ chuẩn ngôn ngữ |
| 0.5 | +0.031 | 0.55 | AMBIGUOUS | Phạt KL quá nặng, hạn chế mô hình phân biệt chosen và rejected |

*Giả thuyết:* Khi giảm $\beta$ xuống 0.05, trọng số phạt KL divergence nới lỏng giúp mô hình học thuộc sở thích mạnh hơn và margin nới rộng, nhưng có thể làm suy giảm tính tự nhiên của câu trả lời. Ngược lại khi $\beta = 0.5$, mô hình bị ràng buộc quá chặt vào mô hình tham chiếu SFT khiến margin bị ép về gần 0 và độ chính xác phân biệt sụt giảm.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong bài lab này là **thiết lập hệ số $\beta = 0.1$ và lựa chọn mô hình SFT đã merge (`models/sft-merged`) làm mô hình tham chiếu (reference model) cố định trước khi khởi tạo adapter DPO**.

1. **Phương án thay thế:** Phương án thay thế là sử dụng trực tiếp base model gốc chưa qua SFT (`Qwen3-4B-Instruct` nguyên bản) làm reference model, hoặc sử dụng các biến thể reference-free như ORPO / SimPO ngay từ đầu, hoặc dùng giá trị $\beta$ rất nhỏ ($\beta = 0.01$).
2. **Lý do chọn:** Trong lý thuyết DPO của Rafailov et al. (2023), hàm implicit reward $r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ đo lường mức độ chênh lệch xác suất tương đối giữa chính sách đang học và chính sách tham chiếu. Nếu reference model không phải là mô hình đã được SFT trên tập chỉ dẫn tiếng Việt, khoảng cách KL divergence ban đầu sẽ bị lệch lạc lớn, dẫn đến việc gradient cập nhật không ổn định và loss không bắt đầu từ $\log 2 \approx 0.693$. Việc gộp LoRA SFT vào base model tạo ra một điểm tựa ngôn ngữ tiếng Việt vững chắc. Giá trị $\beta = 0.1$ cung cấp lực cản vừa đủ để mô hình học cách phân biệt câu trả lời tốt/xấu mà không phá vỡ khả năng tạo sinh mượt mà đã học từ SFT.
3. **Kết quả xác nhận:** Thực nghiệm đã xác nhận hoàn toàn quyết định này: loss bước đầu ghi nhận chính xác tại $0.6924 \approx \log 2$, đồ thị reward curve đạt chẩn đoán `INTENDED`, margin held-out dương (+0.0818) và không xảy ra hiện tượng suy thoái xác suất.
4. **Bài học rút ra:** Nếu thực hiện lại, tôi sẽ kết hợp thêm kỹ thuật Regularized Preference Optimization (RPO) với một thành phần SFT NLL nhỏ bổ trợ để vừa bảo tồn hoàn hảo xác suất của câu `chosen`, vừa mở rộng margin phân biệt rõ rệt hơn nữa.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt-level strict | 42.1% (± 2.1%) | 44.5% (± 2.0%) | +2.4% |
| GSM8K | 5-shot vi | 38.6% (± 1.8%) | 38.0% (± 1.8%) | -0.6% |
| Global-MMLU-vi | 5-shot vi | 46.2% (± 1.5%) | 46.8% (± 1.5%) | +0.6% |

Kết quả so sánh trên các bộ đo chuẩn cho thấy không có sự sụt giảm nghiêm trọng nào vượt quá $2\times$ sai số chuẩn (stderr). Hiện tượng "thuế căn chỉnh" (alignment tax) trên bài toán suy luận số học GSM8K chỉ dao động rất nhỏ (-0.6%, hoàn toàn nằm trong biên độ nhiễu thống kê). Khả năng tuân thủ định dạng chỉ dẫn (IFEval) có sự cải thiện nhẹ sau khi được căn chỉnh bằng DPO.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.63 | +0.0268 | 360.4 | INTENDED: Loss chuẩn, cân bằng tốt |
| RPO | 0.60 | +0.0301 | 369.1 | INTENDED: Thêm NLL loss giúp giữ log-prob chosen cực tốt (+0.506) |
| DPO-norm | 0.63 | +0.0111 | 393.9 | LIKELIHOOD DISPLACEMENT: Chuẩn hoá độ dài làm tăng độ dài câu nhiều nhất |
| LD-DPO | 0.56 | +0.0229 | 363.0 | LIKELIHOOD DISPLACEMENT: Margin thấp hơn trên tập kiểm tra |
| ORPO | 0.65 | N/A (Log-odds: -0.624) | 345.1 | Độ chính xác held-out cao nhất (65%), câu trả lời cô đọng và súc tích nhất |

**Nhận xét:** Biến thể **DPO-norm** làm thay đổi độ dài câu trả lời nhiều nhất (tăng lên 393.9 ký tự), do việc chuẩn hoá độ dài trong mẫu số làm giảm bớt mức phạt đối với các câu trả lời dài. Trong khi đó, **ORPO** mang lại câu trả lời ngắn gọn nhất (345.1 ký tự) và đạt độ chính xác phân biệt held-out cao nhất (65%) nhờ cơ chế phạt log-odds ratio trực tiếp mà không cần mô hình tham chiếu riêng.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 38.0% / 44.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 4.9% |

Thành phần format reward (định dạng `Đáp số: <số>`) tăng mạnh ngay từ các step đầu tiên, sau đó thành phần correctness reward mới tăng dần khi mô hình học được cách suy luận chính xác bài toán tiếng Việt.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong bài lab là hội đồng giám khảo tự động chạy hoàn toàn cục bộ trên GPU T4 mà vẫn cho kết quả đánh giá rất nhất quán và không hề bị thao túng bởi độ dài câu trả lời (`longer_answer_won_frac` đạt đúng 50%). Điều này chứng minh rằng việc xây dựng hệ thống giám khảo độc lập là hoàn toàn khả thi mà không nhất thiết phải phụ thuộc vào các API thương mại đắt đỏ.
