# Báo cáo Day 08 — 2A202603092-HỒ ĐĂNG MẠNH

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm mẫu số và nguồn (file nào).


## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | `person=0.40`; các lớp còn lại `0.30` | Run `t_y26` phát hiện 215 box/8 ảnh Pilot ở ngưỡng 0.25, trung bình 0,82 giây/ảnh. Run `t_y26_c05` còn 139 box/8 ảnh ở ngưỡng 0.50, trung bình 0,73 giây/ảnh. Tôi chọn ngưỡng trung gian để hạn chế dự đoán yếu nhưng vẫn giữ các vật thể cần rà soát. |
| three_wheeler | YOLOE-26s, prompt: `three_wheeler` | `0.25` | Run `t_yoloe` phát hiện 8 box/8 ảnh Pilot, trung bình 0,67 giây/ảnh. Mô hình nhận diện được tuk-tuk và xích lô mà YOLO26 dễ nhầm thành `car` hoặc `truck`. Tuy nhiên, vẫn có box sai cần kiểm tra thủ công. |

Nguồn: `submission/model_log.csv`. Các ngưỡng trên là ngưỡng đã sử dụng khi hợp nhất; chưa có kết quả `threshold_sweep` để khẳng định chúng là tối ưu.

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):

```powershell
.\.venv\Scripts\python.exe scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --iou 0.6 --conf 0.3 --thr "person=0.4,three_wheeler=0.25" --split batch --out work/prelabel/merged_batch/pre.coco.json
```

Kết quả hợp nhất: 480 box/24 ảnh Batch, theo `work/prelabel/merged_batch/pre.coco.json`.

- Phiên bản SAM bạn chọn (SAM 1 / 2.1 / 3) và lý do:

Tôi chọn SAM 2.1 (`sam2.1_b.pt`) vì đã cài đặt và chạy được trên CPU, không cần GPU. Run `t_sam` tạo 16 mask/2 ảnh, trung bình 6,92 giây/ảnh. Mô hình hỗ trợ phân đoạn phương tiện có hình dạng phức tạp bằng điểm dương và điểm âm. Tuy nhiên, vẫn có hiện tượng mask lấn sang người hoặc bỏ sót bánh xe và các chi tiết mảnh. Tôi chưa đo chính xác dung lượng RAM sử dụng.

- Lỗi nào không sửa được bằng đổi ngưỡng? Sửa thế nào?

Lỗi nhận nhầm xe tuk-tuk thành `car` hoặc `truck` không thể khắc phục chỉ bằng việc thay đổi ngưỡng confidence, vì YOLO26 không có lớp riêng phù hợp cho loại phương tiện này. Tôi sử dụng YOLOE với prompt `three_wheeler`, ưu tiên dự đoán YOLOE khi gộp nhãn, sau đó kiểm tra và sửa thủ công trong CVAT.

- Box điểm cao mà sai hoặc điểm thấp mà đúng (ảnh nào):

Trong `img_032`, YOLOE nhận diện đúng xe xích lô với confidence khoảng 0.27, cho thấy điểm thấp không đồng nghĩa với dự đoán sai. Tại `img_002`, YOLOE nhận diện đúng xe tuk-tuk với confidence 0.93, nhưng cũng có các box dự đoán nhầm ở khu vực khác. Tôi chưa có ví dụ được xác nhận đầy đủ về một box có confidence rất cao nhưng sai lớp hoặc sai vị trí.

- Giới hạn: chọn ngưỡng trên chính Gold (4 ảnh) nên số đo máy trên Gold là lạc quan:

Bộ Gold có 44 đối tượng/4 ảnh, nguồn `submission/labels/gold_manual.json`. Nếu dùng Gold để chọn ngưỡng rồi tiếp tục đánh giá trên chính những ảnh đó, số đo Precision/Recall có thể lạc quan. Với mẫu chỉ 4 ảnh, kết quả chưa đủ đại diện cho mọi bối cảnh giao thông.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---:|---:|---|
| accepted | 425 | 91,0% | 425/467 box máy |
| edited | 7 | 1,5% | 7/467 box máy |
| deleted | 35 | 7,5% | 35/467 box máy |
| added | 11 | 2,5% | 11/443 đối tượng sau duyệt |

Nguồn: `submission/qc/review_stats.md`, ngưỡng accepted là IoU ≥ 0.9. Báo cáo đã loại 25 đối tượng nằm quanh các bẫy khỏi bảng thống kê để tránh tính lỗi nhân tạo thành lỗi mô hình.

Trong 35 box bị xóa và 11 box được thêm, có 6 trường hợp đổi lớp: máy đặt box đúng chỗ nhưng gán sai lớp. Vì vậy, số vật thể máy thực sự bỏ sót được báo cáo ước tính là 11 − 6 = 5 đối tượng.

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?

Xét theo số box bị xóa, `three_wheeler` là lớp đáng chú ý nhất: 14/30 box = 46,7% bị xóa, ngoài ra 4/30 box = 13,3% phải chỉnh sửa. Mô hình gặp khó khăn ở tuk-tuk và xích lô, ví dụ `img_004`, `img_013` và `img_032`.

Nếu xét theo tỷ lệ box giữ nguyên, `truck` còn thấp hơn, chỉ 2/6 box = 33,3%, nhưng mẫu của lớp này quá nhỏ để kết luận chắc chắn. Vì vậy, tôi ưu tiên kiểm tra `three_wheeler` do có số lỗi tuyệt đối lớn nhất.

Nguồn: bảng thống kê theo lớp trong `submission/qc/review_stats.md`.

- So với Gold: Precision / Recall của máy (mẫu số: đối tượng Gold):

Kết quả đánh giá trên **4 ảnh Gold với 44 đối tượng**, sử dụng ngưỡng đề xuất từ `threshold_sweep.py`:

- True Positive (TP): 39 đối tượng.
- False Positive (FP): 47 dự đoán.
- False Negative (FN): 5 đối tượng.
- Precision (micro): 39/(39 + 47) = 45,3%.
- Recall (micro): 39/(39 + 5) = 88,6%.
- F1-score (micro): 60,0%.

Kết quả cho thấy mô hình phát hiện đúng 39/44 đối tượng Gold nhưng vẫn có 47 dự đoán sai hoặc dư thừa.

Bộ ngưỡng đánh giá gồm `person=0.45`, `motorbike=0.10`, các lớp còn lại sử dụng ngưỡng mặc định `0.30`.

Các chỉ số được tính từ dự đoán YOLO26 và YOLOE trong `threshold_sweep.csv`, chưa phải kết quả đánh giá riêng file dự đoán sau hợp nhất theo recipe Batch. Do sử dụng chính tập Gold để lựa chọn ngưỡng và đánh giá, kết quả có thể lạc quan và chưa đại diện cho dữ liệu độc lập.

Nguồn: `submission/qc/threshold_sweep.csv`, `submission/qc/threshold_sweep.md` và `submission/labels/gold_manual.json`.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 5/12 = 41,7%, KTC 95% khoảng 19–68%, thấp hơn ngưỡng yêu cầu 80%.

Các loại bẫy bị bỏ lọt:

- `loose`: phát hiện 0/3, bỏ lọt 3/3.
- `missing`: phát hiện 1/3, bỏ lọt 2/3.
- `wrong_class`: phát hiện 2/3, bỏ lọt 1/3.
- `extra`: phát hiện 2/3, bỏ lọt 1/3.

Các ảnh có bẫy chưa phát hiện gồm `img_005` (loose), `img_034` (missing), `img_022` (loose), `img_026` (extra), `img_003` (wrong_class), `img_006` (missing) và `img_009` (loose).

Nguồn: `submission/qc/review_stats.md`.

Tôi nhận thấy mình chưa kiểm tra kỹ đường biên và kích thước box, dẫn đến bỏ lọt cả 3/3 lỗi `loose`. Tôi cũng chưa thực hiện đầy đủ bước tự tìm vật thể bị bỏ sót trước khi tin vào nhãn máy.

- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt?

Tỷ lệ accepted đạt 425/467 = 91,0%, nhưng tỷ lệ bắt bẫy chỉ 5/12 = 41,7%. Hai chỉ số dùng mẫu số khác nhau nên không thể so sánh trực tiếp như cùng một thước đo. Tuy nhiên, việc phần lớn box được giữ nguyên trong khi nhiều bẫy bị bỏ lọt cho thấy nguy cơ tôi quá tin vào gợi ý AI. Tôi cần kiểm tra ảnh gốc độc lập hơn thay vì chỉ sửa những lỗi dễ nhìn thấy.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`):

Gán tay Gold: 60 phút/4 ảnh = 15 phút/ảnh.

Duyệt nhãn AI Batch: 30 phút/24 ảnh = 1,25 phút/ảnh.

Nguồn: `submission/time_log.csv`.

Theo thời gian ghi nhận, lượt duyệt Batch nhanh hơn khoảng 12 lần so với gán tay Gold. Tuy nhiên, đây không phải phép so sánh được kiểm soát giữa hai phương pháp trên cùng tập ảnh. Tôi đã gán Gold trước nên quen thao tác hơn khi duyệt Batch. Ngoài ra, tỷ lệ phát hiện bẫy 41,7% cho thấy thời gian duyệt ngắn có thể đi kèm với việc bỏ sót lỗi.

- Kết luận: có / không / chỉ cho lớp nào, dựa trên số liệu:

Tôi chọn có sử dụng AI làm công cụ hỗ trợ, nhưng không chấp nhận tự động hoàn toàn.

YOLO26 hữu ích trong việc tạo nhãn ban đầu cho `person`, `motorbike` và `car`. YOLOE bổ sung khả năng tìm `three_wheeler`. Tuy nhiên, 35/467 box máy bị xóa (7,5%) và việc chỉ phát hiện 5/12 bẫy (41,7%) cho thấy kết quả vẫn cần được kiểm tra kỹ.

Đối với `three_wheeler` và `truck`, tôi cần tăng mức độ rà soát vì tỷ lệ phải chỉnh sửa hoặc xóa còn đáng kể.

- Ba quy tắc giảm thiên kiến của bạn (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):

1. Kiểm tra ảnh gốc trước khi xem nhãn máy: Tự đếm người và phương tiện trong cả 24 ảnh Batch trước khi so sánh với box AI; ghi lại số vật thể bị bỏ sót.
2. Áp dụng checklist cho từng ảnh: Kiểm tra đủ bốn nhóm lỗi `missing`, `extra`, `wrong_class`, `loose`, đặc biệt chú ý box nhỏ, đối tượng bị che khuất và đường biên box.
3. Đánh giá độc lập sau rà soát: Chạy `review_stats.py`, xem cả tỷ lệ accepted và tỷ lệ bắt bẫy. Nếu tỷ lệ bắt bẫy dưới 80%, tiếp tục rà soát trước khi quyết định Release.

## 5. Mask SAM cho 3 vật

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013.jpg | SAM 2.1 (`sam2.1_b.pt`) | 7 (3 dương, 4 âm) | Phiên bản v2 đã loại bỏ phần lớn người lái và hành khách; không có hàng hóa cần tách. | Mask mái và thân xe khá tốt nhưng bỏ sót phần lớn bánh sau, một số chi tiết khung và mép mái chưa chính xác. | `img_013_tuktuk_v2.jpg` |
| img_004.jpg | SAM 2.1 (`sam2.1_b.pt`) | 6 (3 dương, 3 âm) | Không bao phủ đáng kể người và không có hàng hóa; tập trung vào chiếc tuk-tuk ở giữa. | Mask nhận diện đúng xe, ít lấn sang phương tiện bên cạnh nhưng vùng kính, phần bên trong xe và một số đường biên còn chưa chính xác. | `img_004_tuktuk.jpg` |
| img_032.jpg | SAM 2.1 (`sam2.1_b.pt`) | 6 (3 dương, 3 âm) | Chủ yếu bao phủ xích lô, không có hàng hóa; còn lấn nhẹ sang người đứng cạnh. | Phân đoạn được ghế, bánh và phần lớn khung xe, nhưng còn thiếu chi tiết khung mảnh và một số vùng mask bị lấn. | `img_032_cyclo.jpg` |

Nguồn: `submission/masks/` và các lệnh `sam2_points.py` đã chạy.

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?

Nguyên tắc xác định ranh giới vật thể và tách người khỏi phương tiện giúp tôi đánh giá những vùng mask bị lấn hoặc bị thiếu. Tuy nhiên, cần hướng dẫn cụ thể hơn cách xử lý kính xe, khung kim loại mảnh, bánh xe, vật thể bị che khuất và vùng tiếp xúc giữa người với phương tiện.

Khi thêm điểm âm để loại trừ người trong `img_013`, mask tách người tốt hơn nhưng mất một phần bánh sau. Điều này cho thấy cần đánh giá lại toàn bộ mask sau mỗi lần chỉnh sửa điểm.

- (Tuỳ chọn) Interactor SAM trong CVAT: làm được / không; khác gì offline:

Tôi sử dụng SAM 2.1 offline bằng script `sam2_points.py` trên VS Code. Tôi chưa thử Interactor SAM trực tiếp trong CVAT nên chưa có kết quả thực tế để so sánh. Phương pháp offline cho phép nhập tọa độ điểm dương, điểm âm, lưu kết quả và so sánh nhiều phiên bản mask.

## 6. Calibration (Part 6)

- Số quyết định: 0 quyết định đã ghi, nguồn `submission/qc/calibration_decisions.csv`. File hiện chỉ chứa dòng tiêu đề.

  Quan trọng nhất (dẫn `CAL-xx`): Chưa có quyết định Calibration chính thức để dẫn mã.

- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:

 Những nội dung tôi cần thảo luận và bổ sung gồm quy tắc xử lý xe ba bánh, ngưỡng vật thể nhỏ, cách xác định ranh giới box và cách xử lý người ngồi trên phương tiện.

- Nguyên nhân gốc:

Có ba nhóm nguyên nhân: mô hình nhận sai lớp phương tiện, Guideline chưa đủ rõ đối với một số tình huống khó, và bản thân tôi chưa áp dụng nhất quán checklist khi rà soát. Đây là nhận xét sơ bộ từ thực nghiệm, chưa thay thế biên bản Calibration chính thức.

## 7. QC và quyết định (Part 6)

Mẫu QC: 8 ảnh, gồm 3 ca khó và 5 ảnh chọn ngẫu nhiên, tổng cộng 134 đối tượng. Sử dụng taxonomy v1.0.

Kết quả:
- Lỗi theo đối tượng: 6/134 = 4,5%.
- Ảnh có lỗi: 3/8 = 37,5%.
- Accepted: 326/467 = 69,8%.
- Edited: 36/467 = 7,7%.
- Deleted: 105/467 = 22,5%.
- Added: 51/413 = 12,3%.
- Phát hiện bẫy: 11/12 = 91,7%.

Các lỗi được ghi nhận trên phiếu QC:
- `bo_sot`: 4 lỗi tại `img_005.jpg` và `img_034.jpg`.
- `sai_vi_tri_kich_thuoc`: 2 lỗi tại `img_022.jpg` và `img_034.jpg`.

Giới hạn thống kê: Mẫu chỉ có 8 ảnh, chưa đủ cơ sở chứng minh tỷ lệ lỗi toàn bộ dữ liệu dưới ngưỡng 2%. Cận trên Wilson 95% xấp xỉ 69%.

Quyết định sơ bộ: REWORK.

Lý do: Tỷ lệ ảnh lỗi 37,5% vượt ngưỡng 2%; lỗi bỏ sót chiếm 4/6 lỗi, có dấu hiệu lặp lại.

Hướng khắc phục: Xác minh các lỗi nghi vấn trên CVAT, sửa nhãn nếu xác nhận lỗi thực tế, rà soát đối tượng nhỏ hoặc bị che khuất, cập nhật dữ liệu và kiểm tra QC lại trước khi Release.

## 8. Ghi chú cá nhân

- Dùng AI để làm bài (công cụ nào, việc gì, kiểm tra thế nào):

Tôi sử dụng ChatGPT để hỗ trợ hiểu yêu cầu, hướng dẫn thao tác VS Code và PowerShell, giải thích lỗi cài đặt, phân tích kết quả YOLO26, YOLOE và SAM, đồng thời hỗ trợ viết báo cáo. Tôi kiểm tra các số liệu bằng Terminal, các file JSON/CSV của bài và quan sát trực tiếp ảnh kết quả. Việc gán nhãn và rà soát được thực hiện trên CVAT.

- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)?

Tôi đã hoàn thành Gold Manual trước các lượt chạy pre-label của Day 08. Tuy nhiên, **[CẦN TỰ XÁC NHẬN]** việc có tham khảo gợi ý AI về chính các ảnh Gold trước khi hoàn thành hay không. Không thể xác nhận điều này chỉ từ log chạy chương trình.

- Máy / GPU / CVAT dùng; vướng mắc cài đặt:

Tôi sử dụng Windows, Visual Studio Code, PowerShell, Python 3.11.9 và môi trường `.venv`. PyTorch chạy bằng CPU, môi trường không nhận GPU khả dụng. CVAT chạy trên Docker tại `http://localhost:8080`.

Trong quá trình cài đặt, máy từng thiếu thư viện Python. Khi tải trọng số YOLO, kết nối GitHub có lúc timeout và phải thử lại. YOLOE cần thêm thư viện CLIP và trọng số mô hình. Sau khi hoàn tất cài đặt, các mô hình đã chạy thành công.

- Dùng cache `data/cache/` thay vì tự chạy?

Tôi trực tiếp chạy các script suy luận YOLO26, YOLOE và SAM trên máy. Các lượt chạy đã báo cáo không sử dụng pre-label cache để thay thế kết quả thực nghiệm.

- Lỗi / bug script đã gặp (nếu có):

Tôi từng kiểm tra nhầm đường dẫn `work/tasks/batch`, trong khi 24 ảnh nằm ở `work/tasks/review`. Quá trình tải mô hình có lỗi timeout tạm thời. Khi chạy `validate_labels.py`, chương trình cảnh báo 9 box có chiều cao nhỏ hơn 20 pixel; đây là cảnh báo chất lượng dữ liệu cần rà soát, không phải lỗi làm chương trình dừng.

- Một điều bạn làm khác lần sau:

Tôi sẽ quan sát ảnh gốc và kiểm đếm đối tượng trước khi nhìn các box máy dự đoán. Sau đó sử dụng checklist cố định cho bốn loại lỗi: thiếu, thừa, sai lớp và sai ranh giới. Tôi cũng sẽ dành thêm thời gian kiểm tra các box có vẻ hợp lý để tránh bỏ lọt lỗi `loose`.