"""
run_all_clean.py — รันทุกขั้นตอนประเมินผลแบบไม่ผ่านการ balance ในคำสั่งเดียว
================================================================================
    cd pipeline
    python run_all_clean.py            # RF -> CNN -> Ensemble -> สรุปผล (CNN ใช้ GPU นานสุด)
    python run_all_clean.py --rf-only  # เฉพาะ RF (ไม่กี่นาที) แล้วสรุปผล

ต้องมีไฟล์ที่สกัดไว้แล้วในโฟลเดอร์นี้: features_raw_new.npz (RF) และ cnn_data_manifest_new.json + cnn_features_new/ (CNN)
ผลลัพธ์: RESULTS.md, ตารางใน ../README.md, portfolio_text_filled.txt
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
rf_only = "--rf-only" in sys.argv

steps = ["train_rf_clean_new.py"]
if not rf_only:
    steps += ["train_cnn_clean_new.py", "ensemble_clean_new.py"]
steps += ["make_results_clean.py"]

for s in steps:
    print(f"\n{'=' * 70}\n▶ {s}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, s], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"\n❌ {s} ล้มเหลว (exit {r.returncode}) — แก้แล้วรันใหม่ ขั้นที่ผ่านแล้วไม่ต้องรันซ้ำ "
                 f"(รันเฉพาะไฟล์ที่ล้มเหลว แล้วค่อยรัน make_results_clean.py)")
    print(f"✔ {s} เสร็จใน {(time.time() - t0) / 60:.1f} นาที", flush=True)

print("\n🎉 เสร็จทั้งหมด — เปิด RESULTS.md และ portfolio_text_filled.txt")
