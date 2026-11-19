# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603092 - HỒ ĐĂNG MẠNH

> Chỉ ghi những nội dung thay đổi so với Guideline v1.0, không sao chép lại toàn bộ quy tắc cũ. Các quyết định được bổ sung dựa trên lỗi phân loại phương tiện, cảnh báo vật thể nhỏ và bẫy bị bỏ lọt trong quá trình rà soát nhãn AI.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Vật thể có Bounding Box cao dưới 20 pixel có cần gán nhãn riêng hay sử dụng `ignore_region`? | Theo quyết định Calibration đề xuất, không gán box riêng cho vật thể dưới ngưỡng 20 pixel; chỉ sử dụng `ignore_region` cho cụm vật thể quá nhỏ hoặc dày đặc theo điều kiện của Guideline. | `img_018.jpg` | `CAL-02` |
| 2 | Những câu hỏi mở khác trong mục 6 của Guideline v1.0 đã được giải quyết hết chưa? | Chưa xác nhận. Cần đối chiếu nguyên văn mục 6 của v1.0 và ghi lại riêng các câu hỏi chưa chốt, không tự suy đoán quyết định. | Chưa xác định | — |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_004.jpg` | Xe tuk-tuk ba bánh có hình dạng khác với các phương tiện thông thường. | YOLO26m có thể nhận nhầm xe tuk-tuk thành `car` hoặc `truck`. YOLOE-26s được sử dụng bổ sung với prompt `three_wheeler`. | Xếp xe tuk-tuk vào `three_wheeler` theo taxonomy, kiểm tra lại lớp và Bounding Box trên CVAT. | **R-15 (CAL-01):** Phương tiện ba bánh phù hợp định nghĩa taxonomy phải gán `three_wheeler`, không gán `car` hoặc `truck` chỉ vì có hình dạng tương tự. Bổ sung vào phần quy tắc phân loại phương tiện của v1.0. |
| `img_018.jpg` | Một số vật thể rất nhỏ, box có chiều cao dưới 20 pixel, gây khó khăn khi gán nhãn. | `validate_labels.py` phát cảnh báo về vật thể nhỏ. Đây là cảnh báo kiểm tra dữ liệu, không phải lỗi phân loại của mô hình. | Kiểm tra kích thước vật thể theo ngưỡng Guideline; xử lý những cụm quá nhỏ hoặc dày đặc bằng `ignore_region` khi phù hợp. | **R-16 (CAL-02):** Quy định rõ cách xử lý vật thể dưới ngưỡng 20 pixel và điều kiện sử dụng `ignore_region`. Bổ sung vào phần quy tắc vật thể nhỏ của v1.0. |
| `img_005.jpg` | Bounding Box bao quá rộng, chứa phần nền không thuộc vật thể nhưng vẫn có vẻ hợp lý khi quan sát nhanh. | File pre-label hợp nhất từ YOLO26 và YOLOE đã được `inject_traps.py` chèn bẫy `loose`. Đây là bẫy nhân tạo, không được tính là lỗi gốc của mô hình. | Kiểm tra lại bốn cạnh Bounding Box, chỉnh box ôm sát vật thể và loại bỏ phần nền thừa. | **R-17 (CAL-03):** Khi duyệt nhãn AI phải kiểm tra cả vị trí và kích thước box; không chấp nhận box quá rộng hoặc lệch chỉ vì đúng lớp. Bổ sung vào phần quy tắc vẽ và rà soát Bounding Box của v1.0. |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên, quy định taxonomy và các nguyên tắc gán nhãn ban đầu. | — |
| v1.1 | Bổ sung **R-15** về phân loại xe ba bánh, **R-16** về vật thể nhỏ dưới 20 pixel và **R-17** về kiểm tra Bounding Box quá rộng hoặc lệch. | `img_004.jpg` – `CAL-01`; `img_018.jpg` – `CAL-02`; `img_005.jpg` – `CAL-03`. |