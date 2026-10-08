"""
ensemble_clean_new.py — ประเมิน Ensemble (CNN + RF, soft voting) บน test set จริงที่split ก่อน balance
=====================================================================================
ลำดับรัน (ทำในโฟลเดอร์เดียวกัน):
    python train_rf_clean_new.py     -> rf_test_proba_clean.npy, test_labels_clean.npy
    python train_cnn_clean_new.py    -> cnn_test_proba_clean.npy, cnn_test_labels_clean.npy
    python ensemble_clean_new.py     -> ผล ensemble + confusion matrix

สคริปต์นี้ไม่โหลดโมเดล แค่รวมความน่าจะเป็นที่เซฟไว้ จึงรันได้ในไม่กี่วินาที
และเช็กว่า test set ของ RF กับ CNN เป็นชุดเดียวกันจริง (labels ตรงกันทุกแถว) ก่อนรวมผล
"""
import sys
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

CLASSES = ["covid", "healthy", "symptomatic"]
ALPHA = 0.5   # น้ำหนัก CNN (เท่ากับ app.py: soft voting 50/50)


def load():
    try:
        p_rf = np.load("rf_test_proba_clean.npy")
        p_cnn = np.load("cnn_test_proba_clean.npy")
        y_rf = np.load("test_labels_clean.npy")
        y_cnn = np.load("cnn_test_labels_clean.npy")
    except FileNotFoundError as e:
        sys.exit(f"❌ ไม่พบไฟล์ {e.filename} — รัน train_rf_clean_new.py และ train_cnn_clean_new.py ก่อน")
    return p_rf, p_cnn, y_rf.astype(int), y_cnn.astype(int)


def main():
    p_rf, p_cnn, y_rf, y_cnn = load()

    if len(y_rf) != len(y_cnn) or not np.array_equal(y_rf, y_cnn):
        sys.exit(
            "❌ test set ของ RF กับ CNN ไม่ใช่ชุดเดียวกัน (จำนวนหรือ label ไม่ตรงกัน)\n"
            f"   RF test = {len(y_rf)} | CNN test = {len(y_cnn)}\n"
            "   สาเหตุที่พบบ่อย: manifest ของ CNN ข้ามไฟล์ที่หาไม่เจอ ทำให้ลำดับ/จำนวนไม่ตรงกับ features ของ RF\n"
            "   ห้ามรวมผลจนกว่าจะแก้ ไม่งั้นตัวเลข ensemble ไม่มีความหมาย"
        )
    y = y_rf
    print(f"✅ test set ตรงกัน: n = {len(y):,}  | {dict(zip(CLASSES, np.bincount(y, minlength=3).tolist()))}")

    p_ens = ALPHA * p_cnn + (1 - ALPHA) * p_rf
    results = {"RF": p_rf, "CNN": p_cnn, "CNN+RF (ensemble)": p_ens}

    print(f"\n{'โมเดล':<20}{'Acc':>8}{'Macro-AUC':>11}{'Macro-P':>9}{'Macro-R':>9}{'Macro-F1':>10}")
    for name, p in results.items():
        pred = p.argmax(1)
        print(f"{name:<20}{accuracy_score(y, pred):>8.4f}"
              f"{roc_auc_score(y, p, multi_class='ovr', average='macro'):>11.4f}"
              f"{precision_score(y, pred, average='macro'):>9.4f}"
              f"{recall_score(y, pred, average='macro'):>9.4f}"
              f"{f1_score(y, pred, average='macro'):>10.4f}")

    pred = p_ens.argmax(1)
    cm = confusion_matrix(y, pred)
    print("\n--- Ensemble: classification report ---")
    print(classification_report(y, pred, target_names=CLASSES, digits=3))
    print("--- Ensemble: confusion matrix (row = จริง, col = ทำนาย) ---")
    print(cm)

    print("\n--- Sensitivity / Specificity ต่อคลาส (ensemble) ---")
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        print(f"{c:<12} sensitivity = {tp / (tp + fn):.3f} | specificity = {tn / (tn + fp):.3f}")

    np.savetxt("ensemble_clean_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    print("\n💾 บันทึก ensemble_clean_confusion_matrix.csv")


if __name__ == "__main__":
    main()
