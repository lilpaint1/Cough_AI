# pipeline — จากเสียงดิบถึงโมเดล

ทุกสคริปต์รันจากโฟลเดอร์นี้ (`cd pipeline`) ไฟล์ข้อมูลและโมเดลที่สร้างขึ้นไม่ถูกเก็บใน repo

| ลำดับ | ไฟล์ | ทำอะไร | ผลลัพธ์ |
|---|---|---|---|
| 1 | `preprocessing.py` | อ่าน COUGHVID + metadata, ทำความสะอาด, normalize, ตัดส่วนเงียบ แยกเป็นโฟลเดอร์ตามคลาส | `public_dataset/{covid,healthy,symptomatic}/*.wav` (20,188 ไฟล์) |
| 2 | `rf_extract_new.py` | สกัด 416 ฟีเจอร์ต่อไฟล์สำหรับ Random Forest | `features_raw_new.npz` |
| 3 | `cnn_extract_new.py` | แปลงเสียงเป็น Mel-Spectrogram 128×128 | `cnn_features_new/`, `cnn_data_manifest_new.json` |
| 4 | `dump_min_max_new.py` | ค่า min/max สำหรับ normalize ตอน inference | `cough_min_max_new.json` |
| 5 | `train_rf_clean_new.py`, `train_cnn_clean_new.py` | เทรนโดยแบ่งชุดทดสอบก่อนปรับสมดุล | โมเดล `*_clean` + ความน่าจะเป็นบนชุดทดสอบ |
| 6 | `ensemble_clean_new.py`, `make_results_clean.py` | รวมผล CNN+RF (soft voting 50/50) และสรุป metric | `RESULTS.md`, ตารางใน README |

รันขั้น 5-6 ในคำสั่งเดียว: `python run_all_clean.py` (รายละเอียดใน [`EVALUATION.md`](EVALUATION.md))

`archive_v1/` เก็บสคริปต์ที่ใช้สร้างผลในตารางของ README หลัก (`train_*_new.py`, `ensemble_new.py`, `eval_all_metrics_new.py`) และรุ่นทดลองอื่น ๆ
