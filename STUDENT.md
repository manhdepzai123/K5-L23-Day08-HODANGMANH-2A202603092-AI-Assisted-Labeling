# Thông tin học viên

- Họ và tên: HỒ ĐĂNG MẠNH
- MSHV: 2A202603092
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: manhdepzai123
- Repo bài nộp (URL): https://github.com/manhdepzai123/K5-L23-Day08-HODANGMANH-2A202603092-AI-Assisted-Labeling.git
- Commit chốt bài (hash): f0be6ce8ec67ad88ad2fbadd6326a0b2ee84a7e4
- Máy: Windows, chạy PyTorch trên CPU, không có GPU CUDA/MPS khả dụng trong môi trường thực hành.
- CVAT (Docker): Chạy bằng Docker tại `http://localhost:8080`; phiên bản CVAT cần bổ sung.

## Kết quả tóm tắt

- **Máy – accepted / edited / deleted:** 326/467 = 69,8% / 36/467 = 7,7% / 105/467 = 22,5% (mẫu số: 467 box máy); **added:** 51/413 = 12,3% (mẫu số: 413 đối tượng sau duyệt).
- **So với Gold – Precision / Recall của máy:** 45,3% / 88,6% (Micro-Precision: 39/86 dự đoán; Micro-Recall: 39/44 đối tượng Gold). Kết quả từ `threshold_sweep.csv` trên 4 ảnh Gold ở bộ ngưỡng đề xuất; không phải đánh giá độc lập của recipe hợp nhất Batch.
- **Bắt bẫy:** 11/12 = 91,7%.
- **Thời gian:** Gán tay 60 phút/4 ảnh = 15 phút/ảnh; duyệt có pre-label lần đầu 30 phút/24 ảnh = 1,25 phút/ảnh (chưa tính thời gian sửa lại và QC).
- **Lỗi QC:** 37,5% theo ảnh (3/8 ảnh); 4,5% theo đối tượng (6/134 đối tượng), theo phiếu QC hiện tại.
- **Quyết định:** REWORK — Tỷ lệ ảnh có lỗi 37,5% vượt ngưỡng 2%; lỗi bỏ sót xuất hiện lặp lại, chiếm 4/6 lỗi theo phiếu QC hiện tại.