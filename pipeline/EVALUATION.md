# โปรโตคอลประเมินผล

```
ข้อมูลจริง → แบ่ง test 20% (ไม่แตะ) → แบ่ง val 10% จากที่เหลือ (CNN)
          → balance เฉพาะ train → เทรน → รายงานผลบน test จริง (สัดส่วนคลาสตามธรรมชาติ)
```

เหตุผล: test set ต้องเป็นข้อมูลจริงที่โมเดลไม่เคยเห็นและไม่ผ่านการ balance ใด ๆ
สคริปต์รุ่นแรก ([`archive_v1/`](archive_v1/)) balance ก่อนแบ่ง จึงถูกแทนที่ด้วยสคริปต์ `*_clean_new.py`

## วิธีรัน (บนเครื่องที่มี features / dataset)

คำสั่งเดียว:

```bash
cd pipeline
python run_all_clean.py            # RF -> CNN -> Ensemble -> สรุปผล (CNN ใช้ GPU นานสุด)
python run_all_clean.py --rf-only  # เฉพาะ RF ไม่กี่นาที
```

หรือรันทีละขั้น: `train_rf_clean_new.py` → `train_cnn_clean_new.py` → `ensemble_clean_new.py` → `make_results_clean.py`

ต้องมี `features_raw_new.npz` (RF) และ `cnn_data_manifest_new.json` + `cnn_features_new/` (CNN) ที่สกัดไว้แล้วในโฟลเดอร์นี้

ผลลัพธ์:
- `RESULTS.md` ตาราง metric, sensitivity/specificity ต่อคลาส, confusion matrix (ทุกตัวเลขคำนวณจากไฟล์ผลโดยตรง)
- ตารางผลใน `../README.md` ถูกแทนที่อัตโนมัติ
- `portfolio_text_filled.txt` ข้อความพอร์ตที่ใส่ตัวเลขจริงแล้ว (ไม่ถูก commit)
- โมเดล `cough_rf_model_clean.pkl`, `cough_cnn_model_clean.h5`

สคริปต์จะปฏิเสธรวมผลถ้า test set ของ RF กับ CNN ไม่ตรงกัน (จำนวนหรือ label ต่างกัน)

## นำโมเดลรุ่นใหม่ขึ้นเว็บ

เว็บที่ deploy อยู่โหลดโมเดลจาก Google Drive ผ่าน `RF_MODEL_FILE_ID` / `CNN_MODEL_FILE_ID` (ชื่อไฟล์ใน `app.py` คือ `cough_rf_model.pkl` / `cough_cnn_model.h5`) เพื่อให้ผลที่รายงานตรงกับโมเดลบนเว็บ:

1. อัปโหลด `cough_rf_model_clean.pkl` และ `cough_cnn_model_clean.h5` ขึ้น Google Drive (แชร์แบบ anyone with the link)
2. ใน Vercel → Settings → Environment Variables เปลี่ยน `RF_MODEL_FILE_ID` และ `CNN_MODEL_FILE_ID` เป็น ID ของไฟล์ใหม่
3. Redeploy แล้วลองอัดเสียงทดสอบ (ไม่ต้องแก้โค้ด)

## หมายเหตุ
- metric หลักคือ Macro-F1 และ sensitivity/specificity ต่อคลาส เพราะ test คงสัดส่วนจริง (healthy ประมาณ 74.8% ของข้อมูล ทายว่า Healthy ทุกไฟล์จะได้ Accuracy ≈ 74.8%)
- การแบ่งทำตามไฟล์ ไม่ใช่ตามผู้พูด (ข้อมูลสาธารณะไม่มีรหัสผู้พูด)
