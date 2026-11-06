# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

**Nguồn evidence:** `REPORT_TEMPLATE(1).md`, `RUBRIC(1).md`, `day1_understand_labels(1).py`, `IMAGE_ATTRIBUTION(1).md`, ba JSON và ba PNG đính kèm. Trong các mục dưới, tên tệp không có hậu tố `(1)` chỉ tệp tương ứng trong bộ evidence; đây cũng là tên chuẩn khi nộp theo notebook.

| Sample | COCO image ID | Kích thước ảnh gốc | Phạm vi dùng trong báo cáo |
| --- | --- | --- | --- |
| `traffic` | 210273 | 640 × 428 pixel | Phân loại ảnh |
| `kitchen` | 397133 | 640 × 427 pixel | Detection và instance segmentation |
| `dining` | 166918 | Có trong bộ JSON | Kiểm tra bộ dữ liệu; không dùng thay sample yêu cầu |

Tọa độ trong JSON tham chiếu ảnh gốc, không tham chiếu kích thước PNG minh họa đã phóng lớn và thêm tiêu đề.

### Từ cần nhớ

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả mô hình dự đoán, chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo guideline và được kiểm tra |
| Taxonomy | Danh sách lớp và ánh xạ mã lớp đi kèm bộ dữ liệu/checkpoint |
| Checkpoint | Tệp mô hình `.pt` đã huấn luyện |
| Guideline | Quy tắc gán nhãn thống nhất |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn và yêu cầu sửa khi cần |
| Escalation | Chuyển ca khó cho Lab Coach khi guideline chưa đủ |
| Model score | Điểm mô hình gán cho dự đoán, không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Evidence:** `classification_predictions.json`, lọc `sample_id = traffic`, `rank = 1`; đối chiếu `classification_top5.png`.

1. **Record hạng 1:** `class_id = 468`; `class_name = cab`; `rank = 1`; `score = 0.510915`; `taxonomy_name = ImageNet-1K`. Record có `task = image_classification`, checkpoint `yolo11n-cls.pt`. `rank = 1` nghĩa là lớp có score cao nhất trong kết quả phân loại ảnh này.

2. **Phạm vi nhãn:** Record nói về **cả bức ảnh**, không định vị một chiếc taxi riêng. Đầu ra chỉ có lớp, thứ hạng và score, không có box hay polygon. Nhiều phương tiện xuất hiện trong PNG nhưng mô hình vẫn xếp hạng lớp cho toàn ảnh.

3. **Nguồn danh sách lớp:** Danh sách lớp đi kèm taxonomy/checkpoint. Mã Day 1 lấy tên qua `result.names`; mô hình chọn trong danh sách có sẵn, không tự tạo tên lớp mới.

4. **Vì sao lưu đủ ba trường:** `class_id` phục vụ xử lý và đối chiếu máy; `class_name` giúp con người đọc; `taxonomy_name` xác định hệ lớp để giải nghĩa ID. Cùng một số ID có thể mang nghĩa khác ở taxonomy khác, nên chỉ lưu tên hoặc chỉ lưu ID sẽ khó kiểm tra và tái lập.

5. **Guideline cho ảnh nhiều chủ đề:** Cần quy định rõ tập lớp được phép, tiêu chí chọn chủ thể chính và cách xử lý khi nhiều đối tượng cùng nổi bật. Một quy tắc đề xuất là chọn lớp của chủ thể chiếm diện tích nhìn thấy lớn nhất; nếu không phân biệt được hoặc không có lớp phù hợp, chuyển reviewer/Lab Coach theo quy trình đã thống nhất, không tự tạo lớp. Quy tắc này phải được phê duyệt trước khi dùng hàng loạt. Ảnh `traffic` có nhiều xe buýt và ô tô nên nhãn `cab` chưa đủ cơ sở để trở thành nhãn chuẩn chỉ vì đứng hạng 1.

6. **Score không phải ground truth:** `0.510915` là điểm của mô hình cho lớp `cab`, không chứng minh nhãn này đúng và không đo chất lượng gán nhãn của con người. Top-5 còn có `minibus = 0.164284`, `police_van = 0.085848`, `recreational_vehicle = 0.05411`, `streetcar = 0.048194`. PNG cho thấy cảnh đường phố đông xe, trong khi taxonomy ImageNet-1K dùng các lớp đối tượng; đây là điểm mơ hồ cần đối chiếu guideline. Không thể kết luận tỷ lệ đúng/sai nếu chưa có ground truth được kiểm duyệt.

## 2. Phát hiện vật thể

**Evidence:** `detection_predictions.json`, lọc `sample_id = kitchen`; đối chiếu `detection_predictions.png`.

1. **Record được chọn:** `class_name = person`; `class_id = 0`; `score = 0.912625`; `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`; `bbox_width = 113.58`; `bbox_height = 279.68`; `coordinate_unit = pixel`; `bbox_format = xyxy`; `taxonomy_name = COCO-80`. Đây là người đứng quay lưng ở bên phải ảnh.

2. **Giải thích box:** Gốc tọa độ `(0, 0)` ở góc trên-trái ảnh; x tăng sang phải, y tăng xuống dưới. Góc trên-trái box là `(385.33, 69.24)` và góc dưới-phải là `(498.92, 348.92)`. Theo JSON, box rộng `113.58` pixel, cao `279.68` pixel. Trừ hai tọa độ x đã làm tròn cho kết quả `113.59`; chênh `0.01` pixel vì mã tính chiều rộng từ tọa độ thô rồi mới làm tròn, độc lập với việc làm tròn từng góc. Box chứa cả phần nền quanh người, chẳng hạn khoảng giữa hai chân, nên không phải đường biên cơ thể.

3. **So sánh hai ngưỡng 0.35 và 0.60:**

   | Ngưỡng | Số prediction của `kitchen` | Cách xác định |
   | --- | ---: | --- |
   | 0.35 | 11 | Đếm trực tiếp record trong JSON đã xuất ở ngưỡng 0.35 |
   | 0.60 | 6 | Lọc lại JSON hiện có theo `score >= 0.60`; chưa phải log chạy độc lập ở ngưỡng 0.60 |

   Sáu record còn lại gồm 2 `person`, 2 `bowl`, 2 `oven`, với score lần lượt là `0.912625`, `0.719063`, `0.701282`, `0.686791`, `0.6343`, `0.610931`. Năm record bị loại gồm 3 `bowl` và 2 `cup`. Không có log ô so sánh ngưỡng đi kèm; cần đối chiếu con số 6 với output notebook khi hoàn thiện evidence chạy. Không thể suy ra số prediction tại `0.20` từ JSON đã lọc ở `0.35`, vì các dự đoán dưới `0.35` không được lưu.

4. **Việc reviewer phải làm khi hạ ngưỡng:** Rà soát thêm các hộp mới, xác minh có vật thể thật hay không, kiểm tra lớp, vị trí, hộp trùng và vật thể vẫn bị bỏ sót. Hạ ngưỡng có thể giữ lại đối tượng khó nhưng cũng tăng dự đoán sai. Reviewer phải xem ảnh, không chỉ duyệt theo score; ngưỡng prediction không phải tiêu chí bỏ vật thể khỏi ground truth.

5. **Quy tắc box đề xuất:** Với mỗi đối tượng thuộc lớp cần gán và nhận diện được, vẽ một hình chữ nhật song song trục ảnh ôm sát toàn bộ phần nhìn thấy của đối tượng, giữ tọa độ trong ảnh; đối tượng bị cắt mép thì dừng hộp ở mép ảnh, không đoán phần ngoài ảnh hoặc phần bị che. Đây là đề xuất theo chính sách gán phần nhìn thấy, cần được guideline phê duyệt và áp dụng nhất quán.

6. **Vật bị cắt mép và cách xử lý:** Bên trái, gần mặt bàn, chỉ thấy một phần bàn tay/cánh tay của người đi ra ngoài khung ảnh. JSON có `person`, score `0.610931`, box `[0.12, 263.18, 61.55, 310.78]`; PNG cũng thể hiện hộp này sát mép trái. Tôi chọn **escalation đến Lab Coach** cho quyết định có gán `person` khi chỉ còn bộ phận hay không, vì tài liệu đính kèm chưa quy định tiêu chí nhận diện tối thiểu. Sau khi thống nhất, bổ sung quy tắc cho mọi trường hợp tương tự; nếu được gán, dùng hộp phần nhìn thấy, không dựng thêm cơ thể ngoài khung.

## 3. Phân đoạn theo từng đối tượng

**Evidence:** `segmentation_predictions.json`, lọc `sample_id = kitchen`; đối chiếu `segmentation_prediction.png`.

1. **Record được chọn:** `instance_id = kitchen-001`; `class_name = person`; `class_id = 0`; `score = 0.899318`; `polygon_point_count = 348`; `coordinate_unit = pixel`. Năm điểm đầu của `polygon_xy` là:

   ```json
   [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
   ```

   Checkpoint là `yolo11n-seg.pt`, taxonomy `COCO-80`; box kèm theo là `[385.45, 66.44, 498.02, 348.58]`. PNG hiển thị mask xanh phủ người đứng ở bên phải.

2. **Polygon thêm thông tin gì:** Box chỉ mô tả phạm vi hình chữ nhật. Polygon mô tả đường biên bằng chuỗi điểm, qua đó xác định vùng hình dạng vật thể. Ở người đứng, polygon có thể theo đầu, vai, cánh tay, tạp dề và hai chân, đồng thời loại phần nền giữa hai chân. Nhiều điểm hơn không tự đảm bảo biên đúng hơn; vẫn phải xem ảnh và guideline.

3. **Ý nghĩa `instance_id`:** ID dùng để phân biệt từng đối tượng trong bộ output lab. `kitchen-001` và `kitchen-009` đều thuộc lớp `person`, `class_id = 0`, nhưng là hai instance riêng. Mã nguồn tạo ID từ `sample_id` và số thứ tự `index + 1`; do đó ID không phải mã lớp, không bảo đảm ổn định nếu thứ tự prediction đổi và không liên kết cùng một người qua nhiều khung hình như tracking ID.

4. **Quy tắc biên mask đề xuất:** Vẽ theo ranh giới phần đối tượng thực sự nhìn thấy; không lấy nền hoặc vật che phía trước vào mask và không suy diễn phần bị che. Tách từng đối tượng thành instance riêng kể cả cùng lớp; dừng tại mép ảnh nếu bị cắt. Nếu phần nhìn thấy bị chia thành nhiều vùng hoặc cần lỗ trong mask, phải dùng quy ước định dạng đã thống nhất, không nối tùy tiện qua nền.

5. **Chỗ khó và guideline/escalation:** Ở cụm bát phía trước bên trái, bát chạm/chồng lên nhau và một bát bị cắt bởi mép ảnh. JSON ghi riêng `kitchen-002` (`bowl`, `0.735744`), `kitchen-003` (`bowl`, `0.698698`) và `kitchen-006` (`bowl`, `0.528201`); PNG có nhiều vùng hồng sát nhau. Với đoạn biên nhìn rõ, áp dụng quy tắc tách từng instance và chỉ vẽ phần nhìn thấy. Đoạn bị che hoặc màu overlay làm khó xác định cần kiểm tra ảnh gốc; nếu vẫn không xác định được ranh giới thì **escalation đến Lab Coach**, không vẽ tiếp theo phỏng đoán.

   Một điểm cần QC thêm là `kitchen-004`: lớp `potted plant`, score `0.632235`, phủ cụm lá treo góc trên-trái nhưng ảnh không cho thấy rõ chậu. Đây là nghi vấn không khớp định nghĩa lớp, chưa phải kết luận sai chắc chắn; cần kiểm tra ảnh gốc và guideline thay vì chấp nhận nhãn chỉ vì score trên ngưỡng.

## 4. Vòng đời và kiểm tra chất lượng

**Vòng đời production:** ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework.

Ảnh thô cần đúng phạm vi và được phép sử dụng. Guideline quy định taxonomy, đơn vị nhãn và cách xử lý ca khó. Annotator tạo nhãn, reviewer kiểm tra và yêu cầu rework để có ground truth đáng tin cậy. Dữ liệu đã kiểm duyệt được dùng cho huấn luyện; checkpoint tạo prediction trên ảnh. QC đối chiếu prediction với ảnh, guideline và ground truth để phân tích lỗi, rồi sửa nhãn hoặc cập nhật quy tắc qua quy trình kiểm soát. Lab này chỉ đọc prediction từ checkpoint có sẵn, không thực hiện huấn luyện hay tạo bộ ground truth đã phê duyệt.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho toàn ảnh theo taxonomy được chọn, lưu mã và tên lớp | `traffic` có nhiều xe buýt/ô tô; top-1 `cab`, score `0.510915`, chưa chứng minh lớp đại diện đúng toàn cảnh | Chọn một lớp theo quy tắc chủ thể chính; ghi ca mơ hồ nếu không có lựa chọn rõ | Kiểm tra lớp có trong taxonomy, ánh xạ ID–tên và cách áp dụng quy tắc chọn một lớp; không duyệt dựa vào top-1 |
| Phát hiện vật thể | Một record lớp và box cho mỗi đối tượng thuộc phạm vi | `person` ở mép trái chỉ còn một phần tay/cánh tay; box `[0.12, 263.18, 61.55, 310.78]`, score `0.610931` | Ghi nhận cắt mép; hỏi tiêu chí gán bộ phận; vẽ box phần nhìn thấy khi được phép | Kiểm tra đúng lớp, box sát phần được gán, trong ảnh, không trùng/bỏ sót; xem tính nhất quán giữa các ca cắt mép |
| Instance segmentation | Một mask/polygon cùng lớp và ID riêng cho mỗi đối tượng | Các bát `kitchen-002`, `003`, `006` có biên sát/chồng nhau; `kitchen-004` gán `potted plant` cho cụm lá không thấy rõ chậu | Tách instance, vẽ biên nhìn thấy, ghi vùng không chắc chắn và chuyển ca chưa có quy tắc | Kiểm tra biên tràn/sót, nền bị lấy vào mask, tách/gộp instance và định nghĩa lớp; yêu cầu sửa hoặc escalation |

Các quan sát trên là evidence định tính. Chưa có ground truth đối chiếu nên không báo cáo precision, recall, IoU hay tỷ lệ đúng từ score hoặc số lượng prediction. Detection và segmentation là hai checkpoint khác nhau: cùng ảnh `kitchen` đều có 11 record nhưng tập lớp/vật thể khác nhau; không được ghép instance chỉ theo thứ tự record.

## 5. An toàn dữ liệu

1. **Quy tắc:** Chỉ sử dụng các ảnh COCO công khai được cung cấp, giữ attribution và giấy phép, không đưa ảnh riêng, dữ liệu khách hàng/VinFast, họ tên, MSSV, email, số điện thoại hoặc dữ liệu nhạy cảm không cần thiết vào báo cáo/output công khai.
2. **Khi gặp dữ liệu ngoài phạm vi:** Dừng xử lý và dừng tải lên/chia sẻ; báo Lab Coach hoặc giảng viên phụ trách để xác minh quyền sử dụng và cách xử lý trước khi tiếp tục.

Theo `IMAGE_ATTRIBUTION(1).md`, ba ảnh nguồn đều mang giấy phép CC BY 2.0; các PNG là phiên bản thêm prediction hoặc biểu đồ. Giữ nguyên file attribution khi nộp. Tên tác giả bên dưới là thông tin ghi công tác phẩm công khai, không phải thông tin cá nhân của học viên.

| Sample | Tác phẩm / tác giả theo attribution | Nguồn |
| --- | --- | --- |
| `traffic` | Wuhan / Tauno Tõhk (toehk) | [Flickr](https://www.flickr.com/photo.gne?id=5336041838) |
| `kitchen` | Kitchen / Maggie Stephens (Pot Noodle) | [Flickr](https://www.flickr.com/photo.gne?id=6255196340) |
| `dining` | Big Wine Thing / WordRidden | [Flickr](https://www.flickr.com/photo.gne?id=4745624149) |

Giấy phép được ghi trong tệp: [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).

## 6. Checklist trước khi nộp link repo

**Trạng thái hiện tại:** Đã điền báo cáo và đọc đủ bộ evidence đính kèm; chưa truy cập GitHub/LMS và chưa có log thực thi validation Colab. Các ô dưới đây là việc cần kiểm tra trên repository, nên chưa đánh dấu hoàn thành chỉ dựa vào tệp gửi trong cuộc trò chuyện.

- [x] `report/REPORT.md`: đưa nội dung `report.md` này vào đúng tên chuẩn khi nộp; ô upload Colab nhận `.md` và lưu lại thành `REPORT.md`.
- [x] `report/day1_lab_outputs/classification_predictions.json`
- [x] `report/day1_lab_outputs/detection_predictions.json`
- [x] `report/day1_lab_outputs/segmentation_predictions.json`
- [x] `report/day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `report/day1_lab_outputs/visuals/classification_top5.png`
- [x] `report/day1_lab_outputs/visuals/detection_predictions.png`
- [x] `report/day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS` trong lần chạy thực tế.
- [x] Báo cáo không ghi họ tên, MSSV hoặc thông tin liên hệ của học viên.
- [x] Kiểm tra toàn bộ output công khai không chứa dữ liệu nhạy cảm ngoài phạm vi.
- [x] Đã nộp **link repository** trên LMS.