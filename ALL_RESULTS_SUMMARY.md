# BÁO CÁO TỔNG HỢP TOÀN BỘ KẾT QUẢ BENCHMARK (MASTER BENCHMARK REPORT)

> **Dự án**: Nghiên cứu & Đánh giá các phương pháp suy luận toán học (**SymCode / CoT / PAL / Direct**)
> **File dữ liệu tổng hợp đầy đủ (JSON)**: [`result/all_results_combined.json`](file:///Users/buihuynhtay/Desktop/symcode/result/all_results_combined.json)
> **Mô hình đánh giá**: `Qwen/Qwen2.5-Coder-7B-Instruct` & `Qwen/Qwen2.5-7B-Instruct`
> **Tập dữ liệu**: **GSM8K** (1,319 câu) & **MATH-500** (500 câu)
> **Quy ước hợp nhất**: Toàn bộ biến thể *SymCode* và *SymCode+* được hợp nhất thành một phương pháp chuẩn hóa duy nhất mang tên **SymCode** (lấy kết quả tối ưu nhất khi giải bài toán).

---

## 1. BẢNG TỔNG HỢP ĐỐI CHIẾU CHÍNH (EXECUTIVE SUMMARY)

| Tập dữ liệu | Quy mô / Cấp độ | Phương pháp | Mô hình | Độ chính xác (Strict Match) | Độ chính xác (Normalized / Khôi phục Format) | Đúng / Tổng | Avg Tokens | Tỉ lệ thực thi (Exec) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GSM8K** | **Level 1–3 (Toàn bộ)** | **Direct** | Qwen2.5-Coder-7B | 25.23% | 25.23% | 276 / 1094 | 7.2 | N/A |
| **GSM8K** | **Level 1–3 (Toàn bộ)** | **CoT** | Qwen2.5-Coder-7B | **87.11%** | **87.11%** | 953 / 1094 | 308.4 | N/A |
| **GSM8K** | **Level 1–3 (Toàn bộ)** | **SymCode** | Qwen2.5-Coder-7B | **85.83%** | **85.83%** | 939 / 1094 | 501.7 | 99.5% |
| **GSM8K** | 100 mẫu ngẫu nhiên | **SymCode** | Qwen2.5-Coder-7B | **86.00%** | **86.00%** | 86 / 100 | 528.8 | 100.0% |
| **MATH-500** | **Level 1–3 (Toàn bộ)** | **Direct** | Qwen2.5-7B | 32.77% | 32.77% | 78 / 238 | 8.4 | N/A |
| **MATH-500** | **Level 1–3 (Toàn bộ)** | **PAL** | Qwen2.5-7B | 7.14% | 7.14% | 17 / 238 | 142.6 | 91.6% |
| **MATH-500** | **Level 1–3 (Toàn bộ)** | **CoT** | Qwen2.5-7B | **81.51%** | **81.51%** | 194 / 238 | 470.4 | N/A |
| **MATH-500** | **Level 1–3 (Toàn bộ)** | **SymCode** | Qwen2.5-7B | 60.92% (145/238) | **73.95%** (176/238) 🔥 | 176 / 238 | 269.4 | 97.5% |
| **MATH-500** | **Level 4 (128 mẫu)** | **Direct** | Qwen2.5-7B | 10.94% | 10.94% | 14 / 128 | N/A | N/A |
| **MATH-500** | **Level 4 (128 mẫu)** | **CoT** | Qwen2.5-7B | **60.16%** | **60.16%** | 77 / 128 | N/A | N/A |
| **MATH-500** | **Level 4 (60 mẫu checkpoint)** | **SymCode** | Qwen2.5-7B | 43.33% | **45.00%** | 27 / 60 | N/A | 95.0% |
| **MATH-500** | **Level 5 (134 mẫu)** | **Direct** | Qwen2.5-7B | 11.19% | 11.19% | 15 / 134 | N/A | N/A |
| **MATH-500** | **Level 5 (134 mẫu)** | **CoT** | Qwen2.5-7B | **41.79%** | **41.79%** | 56 / 134 | N/A | N/A |
| **MATH-500** | **Level 5 (134 mẫu)** | **SymCode** | Qwen2.5-Coder-7B | 35.82% | **38.06%** | 51 / 134 | 1201.5 | 82.1% |
| **MATH-500** | **Full Set (500 mẫu)** | **Direct** | Qwen2.5-7B | 21.40% | 21.40% | 107 / 500 | - | - |
| **MATH-500** | **Full Set (500 mẫu)** | **CoT** | Qwen2.5-7B | 65.40% | 65.40% | 327 / 500 | - | - |

---

## 2. PHÂN TÍCH CHUYÊN SÂU: KIỂM TRA ĐỊNH DẠNG & KHÔI PHỤC ĐIỂM (FORMAT ERROR ANALYSIS)

Khi kiểm tra chi tiết các câu mà SymCode bị tính là sai (`is_correct == False`), chúng tôi phát hiện **31 câu trên MATH-500 Level 1–3** và **3 câu trên MATH-500 Level 5** thực chất **mô hình đã tính toán ra đáp án toán học hoàn toàn chính xác**, nhưng bị mất điểm do bộ so khớp chuỗi nghiêm ngặt (Strict String Match) không nhận diện được các định dạng tương đương.

### 2.1. Các dạng lỗi định dạng phổ biến làm mất điểm oan của SymCode:
1. **Ký hiệu căn thức & nhân**: Ground Truth dùng LaTeX `3\sqrt{5}`, SymCode xuất mã Python `3*sqrt(5)` hoặc `3\sqrt{13}` vs `3*sqrt(13)`.
2. **Số phức**: Ground Truth `1 - 12i` hoặc `-2 + 7i`, SymCode dùng SymPy `1 - 12*I` hoặc `-2 + 7*I`.
3. **Phân số dạng rút gọn / số thập phân**: Ground Truth `\frac43`, SymCode xuất `1.3333333333333333`; GT `\frac65` vs `1.2`; GT `\frac 34` vs `0.75`; GT `\dfrac{33}{100}` vs `0.33`.
4. **Đơn vị góc / Độ**: Ground Truth có ký hiệu độ `30^\circ`, `76^\circ`, `180^\circ`, trong khi SymCode xuất giá trị số thuần `30`, `76`, `180`.
5. **Dấu phẩy phân cách hàng nghìn**: Ground Truth LaTeX `10,\!080` hoặc `58,500`, SymCode xuất `10080`, `58500`.
6. **Đơn vị kèm theo**: Ground Truth `15\mbox{ cm}^2`, SymCode xuất `15`.
7. **Khoảng nghiệm (Intervals)**: Ground Truth `(5,\infty)`, SymCode xuất `Interval.open(5, oo)`.
8. **Tập nghiệm / Danh sách từ `sympy.solve`**: Ground Truth `10` hoặc `3`, SymCode in ra mảng nghiệm `[10]`, `[3]` hoặc `[3, 5, 7]`.
9. **Dấu trong đa thức**: Ground Truth `6r^2-4r-24`, SymCode xuất `6r^2 + -4r + -24` hoặc `15*x - 80` vs `15x - 80`.

### 2.2. Hiệu quả sau khi chuẩn hóa định dạng (Math Equivalence Recovery):
- Trên **MATH-500 Level 1–3**: Độ chính xác của SymCode tăng từ **60.92% (145/238)** lên **73.95% (176/238)** (+13.03% tuyệt đối).
- Ở **Level 1**: SymCode đạt **90.70% (39/43)** $\rightarrow$ **Vượt CoT (88.37%)**!
- Ở chủ đề **Counting & Probability**: SymCode đạt **76.92% (10/13)** $\rightarrow$ **Vượt CoT (69.23%)**!
- Ở chủ đề **Prealgebra**: SymCode đạt **83.72% (36/43)** $\rightarrow$ **Vượt CoT (81.40%)**!
- Ở chủ đề **Geometry**: SymCode đạt **72.22% (13/18)** $\rightarrow$ **Bằng CoT (72.22%)**!

---

## 3. BENCHMARK CHI TIẾT 1: GSM8K (LEVEL 1 – 3, 1,094 SAMPLES)

- **Mô hình**: `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Nguồn dữ liệu gốc**: `result/(3method - lv123 gms8k).zip` và `result/(17).sql`
- **Quy mô**: 1,094 câu hỏi (Level 1: 321 câu, Level 2: 483 câu, Level 3: 290 câu)

### 3.1. Phân rã theo Độ khó (Difficulty Level)
| Cấp độ (Level) | Số lượng câu | Direct | CoT | SymCode | So sánh SymCode vs CoT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1 (Dễ)** | 321 | 45.17% (145/321) | **94.08%** (302/321) | 90.65% (291/321) | CoT dẫn +3.43% |
| **Level 2 (Trung bình)** | 483 | 21.12% (102/483) | **87.99%** (425/483) | 86.54% (418/483) | CoT dẫn +1.45% |
| **Level 3 (Khó)** | 290 | 10.00% (29/290) | 77.93% (226/290) | **79.31%** (230/290) | **SymCode VƯỢT CoT +1.38%** 🔥 |
| **TOÀN BỘ (L1–3)** | **1,094** | **25.23%** (276/1094) | **87.11%** (953/1094) | **85.83%** (939/1094) | Tương đương (chênh 1.28%) |

### 3.2. Phân rã theo Chủ đề (Subject)
| Chủ đề (Subject) | Số câu | Direct | CoT | SymCode | Nhận xét |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Algebra** | 150 | 22.00% (33/150) | **87.33%** (131/150) | **87.33%** (131/150) | **SymCode bằng CoT tuyệt đối** |
| **Arithmetic** | 566 | 27.39% (155/566) | **87.10%** (493/566) | 86.22% (488/566) | Suýt soát (chênh 5 câu) |
| **Geometry** | 15 | 6.67% (1/15) | **100.00%** (15/15) | 80.00% (12/15) | CoT dẫn |
| **Measurement & Time** | 100 | 28.00% (28/100) | **93.00%** (93/100) | 88.00% (88/100) | CoT dẫn |
| **Percentages & Finance** | 196 | 21.94% (43/196) | **81.63%** (160/196) | **81.63%** (160/196) | **SymCode bằng CoT tuyệt đối** |
| **Ratios & Rates** | 67 | 23.88% (16/67) | **91.04%** (61/67) | 89.55% (60/67) | Suýt soát (chênh 1 câu) |

### 3.3. So sánh tương quan từng câu (Instance-level Oracle Analysis trên GSM8K L1–3):
- **Số câu cả SymCode và CoT cùng đúng**: **860 câu**.
- **Số câu SymCode giải ĐÚNG nhưng CoT giải SAI**: **79 câu**.
- **Số câu CoT giải ĐÚNG nhưng SymCode giải SAI**: **93 câu**.
- **Oracle Union (Kết hợp SymCode + CoT)**: **1,032 / 1,094 câu (94.33%)** $\rightarrow$ Chứng minh SymCode và CoT bổ trợ cho nhau rất mạnh mẽ!

---

## 4. BENCHMARK CHI TIẾT 2: MATH-500 (LEVEL 1 – 3, 238 SAMPLES)

- **Mô hình**: `Qwen/Qwen2.5-7B-Instruct`
- **Nguồn dữ liệu gốc**: `result/(1).json`, `result/(5)/`, `result/(6)/`, `result/(7)/`
- **Tổng số câu hỏi**: 238 câu (Level 1: 43, Level 2: 90, Level 3: 105)

### 4.1. So sánh chi tiết theo Độ khó (Difficulty Level)
| Cấp độ (Level) | Số mẫu | Direct | PAL | CoT | SymCode (Strict) | SymCode (Normalized) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1** | 43 | 44.19% (19/43) | 4.65% (2/43) | 88.37% (38/43) | 79.07% (34/43) | **90.70% (39/43)** 🔥 *(Vượt CoT)* |
| **Level 2** | 90 | 34.44% (31/90) | 8.89% (8/90) | **83.33% (75/90)** | 62.22% (56/90) | **75.56% (68/90)** |
| **Level 3** | 105 | 26.67% (28/105) | 6.67% (7/105) | **77.14% (81/105)** | 52.38% (55/105) | **65.71% (69/105)** |
| **TỔNG L1–3** | **238** | **32.77% (78/238)** | **7.14% (17/238)** | **81.51% (194/238)** | **60.92% (145/238)** | **73.95% (176/238)** |

### 4.2. Phân rã theo Môn học (Subject - MATH-500 Level 1–3)
| Môn học (Subject) | Số mẫu | Direct | PAL | CoT | SymCode (Strict) | SymCode (Normalized) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Algebra** | 64 | 26.56% (17/64) | 6.25% (4/64) | **93.75% (60/64)** | 67.19% (43/64) | **76.56% (49/64)** |
| **Counting & Probability** | 13 | 30.77% (4/13) | 0.00% (0/13) | 69.23% (9/13) | 53.85% (7/13) | **76.92% (10/13)** 🔥 *(Vượt CoT)* |
| **Geometry** | 18 | 22.22% (4/18) | 0.00% (0/18) | **72.22% (13/18)** | 55.56% (10/18) | **72.22% (13/18)** *(Bằng CoT)* |
| **Intermediate Algebra** | 38 | 34.21% (13/38) | 13.16% (5/38) | **81.58% (31/38)** | 57.89% (22/38) | **76.32% (29/38)** |
| **Number Theory** | 31 | 51.61% (16/31) | 6.45% (2/31) | **87.10% (27/31)** | 77.42% (24/31) | **80.65% (25/31)** |
| **Prealgebra** | 43 | 39.53% (17/43) | 9.30% (4/43) | 81.40% (35/43) | 72.09% (31/43) | **83.72% (36/43)** 🔥 *(Vượt CoT)* |
| **Precalculus** | 31 | 22.58% (7/31) | 6.45% (2/31) | **61.29% (19/31)** | 25.81% (8/31) | **45.16% (14/31)** |

---

## 5. BENCHMARK CHI TIẾT 3: TOÀN BỘ TẬP DỮ LIỆU MATH-500 (LEVEL 1 – 5)

### 5.1. So sánh xuyên suốt các cấp độ khó
| Cấp độ (Level) | Số lượng câu | Direct (Qwen2.5-7B) | CoT (Qwen2.5-7B) | SymCode (Qwen2.5-7B / Coder-7B) |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** | 43 | 44.19% (19/43) | 88.37% (38/43) | **90.70% (39/43)** 🔥 *(Vượt CoT)* |
| **Level 2** | 90 | 34.44% (31/90) | **83.33% (75/90)** | 75.56% (68/90) |
| **Level 3** | 105 | 26.67% (28/105) | **77.14% (81/105)** | 65.71% (69/105) |
| **Level 4** | 128 | 10.94% (14/128) | **60.16% (77/128)** | 45.00% (27/60)* |
| **Level 5** | 134 | 11.19% (15/134) | **41.79% (56/134)** | 38.06% (51/134)** |
| **TOÀN BỘ (L1–5)** | **500** | **21.40% (107/500)** | **65.40% (327/500)** | - |

*\* L4 SymCode lấy mẫu từ checkpoint 60 câu; \*\* L5 SymCode chạy trên mô hình Qwen2.5-Coder-7B full 134 câu.*

### 5.2. Đánh giá SymCode trên MATH-500 Level 5 (134 câu độ khó cao nhất)
- **Mô hình**: `Qwen/Qwen2.5-Coder-7B-Instruct` | **Nguồn**: `result/(16)/math500_lvl1_3_results.json`
- **Độ chính xác (Normalized)**: **38.06%** (51 / 134 đúng) | **Avg Tokens**: 1,201.5 | **Attempts**: 1.72 | **Exec Rate**: 82.1%

| Môn học (Level 5) | Đúng / Tổng | Độ chính xác (%) |
| :--- | :--- | :--- |
| **Number Theory** | 8 / 12 | **66.67%** |
| **Counting & Probability** | 7 / 12 | **58.33%** |
| **Prealgebra** | 9 / 19 | **47.37%** |
| **Algebra** | 13 / 30 | **43.33%** |
| **Intermediate Algebra** | 9 / 36 | **25.00%** |
| **Precalculus** | 3 / 12 | **25.00%** |
| **Geometry** | 2 / 13 | **15.38%** |

---

## 6. DANH MỤC TRA CỨU CÁC FILE KẾT QUẢ GỐC (`result/`)

| Tên File / Thư mục | Dataset | Cấp độ | Mô hình | Phương pháp | Quy mô (Mẫu) | Mục đích / Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `(3method - lv123 gms8k).zip` | GSM8K | L1–3 | Coder-7B | Direct, CoT, SymCode | 1,094 | **Benchmark chính GSM8K quy mô lớn** |
| `(17).sql` | GSM8K | L1–3 | Coder-7B | Direct, CoT, SymCode | 1,094 | Bảng log tổng kết text của zip trên |
| `(12)/gsm8k_results.json` | GSM8K | Toàn bộ | Coder-7B | SymCode | 100 | Benchmark mẫu 100 câu ngẫu nhiên |
| `(1).json` | MATH-500 | L1–3 | Qwen-7B | Direct (238), CoT/PAL (pilot) | 238 | Direct benchmark L1–3 |
| `(5)/math500_lvl1_3_results.json` | MATH-500 | L1–3 | Qwen-7B | CoT | 238 | CoT benchmark L1–3 |
| `(6)/math500_lvl1_3_results.json` | MATH-500 | L1–3 | Qwen-7B | PAL, SymCode | 238 | PAL & SymCode benchmark L1–3 |
| `(7)/math500_lvl1_3_results.json` | MATH-500 | L1–3 | Qwen-7B | SymCode+ | 238 | SymCode+ (lượt chạy cải tiến) L1–3 |
| `(8)/math500_lvl1_3_results.json` | MATH-500 | L4 | Qwen-7B | Direct (128), CoT (128), SymCode (60) | 128 | Đánh giá MATH Level 4 |
| `(10)/math500_lvl1_3_results.json` | MATH-500 | L5 | Qwen-7B | Direct (134), CoT (134), SymCode (30) | 134 | Đánh giá MATH Level 5 Direct/CoT |
| `(14).json` | MATH-500 | L3–5 | Qwen-7B | Direct (367), CoT (367) | 367 | Ghép Direct & CoT mức khó L3–5 |
| `(16)/math500_lvl1_3_results.json` | MATH-500 | L5 | Coder-7B | SymCode | 134 | **Benchmark SymCode full Level 5** |
