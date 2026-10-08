"""
train_rf_clean_new.py — RF เวอร์ชันsplit ก่อน balance (split ก่อน → SMOTE เฉพาะ train)
ผลลัพธ์: cough_rf_model_clean.pkl, rf_test_proba_clean.npy, test_labels_clean.npy
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import threading
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
DATA_PATH = "features_raw_new.npz"
MODEL_OUTPUT_PATH = "cough_rf_model_clean.pkl"
CLASSES = ["covid", "healthy", "symptomatic"]

# ─── Hyperparameters ────────────────────────────────────────────────────────
N_ESTIMATORS     = 600
MAX_DEPTH        = 25
MAX_FEATURES     = "sqrt"
MIN_SAMPLES_LEAF = 2
N_JOBS           = -1
# ────────────────────────────────────────────────────────────────────────────


def smote_with_progress(X, y):
    """
    รัน SMOTE บน background thread พร้อม spinner + elapsed time
    เพราะ SMOTE ไม่รองรับ callback → ใช้ thread แสดง progress แทน
    """
    result = {}
    exc_holder = {}

    def _run():
        try:
            sm = SMOTE(random_state=42)
            result['X'], result['y'] = sm.fit_resample(X, y)
        except Exception as e:
            exc_holder['err'] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    idx = 0
    t0 = time.time()
    while t.is_alive():
        elapsed = time.time() - t0
        print(f"\r   {spinner[idx % len(spinner)]}  SMOTE กำลังทำงาน... "
              f"{elapsed:.1f}s  (สร้าง synthetic samples สำหรับ {len(X):,} rows × {X.shape[1]} features)",
              end="", flush=True)
        idx += 1
        time.sleep(0.15)

    print()  # newline หลัง spinner หยุด

    if 'err' in exc_holder:
        raise exc_holder['err']

    elapsed = time.time() - t0
    print(f"   ✅ SMOTE เสร็จใน {elapsed:.1f}s")
    return result['X'], result['y']


def train_rf_model():
    print("🔄 กำลังโหลดข้อมูลสำหรับ RF...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ ไม่พบไฟล์ {DATA_PATH} โปรดรัน rf_extract.py ก่อน")
        return

    data = np.load(DATA_PATH)
    X = data["X"]
    y = data["y"]
    print(f"   → X shape: {X.shape} | classes: {np.unique(y, return_counts=True)}")

    # ✅ CLEAN: แบ่ง train/test "ก่อน" แล้วค่อย SMOTE เฉพาะ train
    #    (เดิม SMOTE ทั้งก้อนก่อนแบ่ง → test มีตัวอย่างสังเคราะห์ที่ลอกจาก train = การรั่วของข้อมูล test)
    print("✂️  แบ่งข้อมูล train/test (80/20) บนข้อมูลจริง ก่อน balance...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"   → train: {np.unique(y_train, return_counts=True)}")
    print(f"   → test (จริง ไม่บาลานซ์): {np.unique(y_test, return_counts=True)}")

    print("⚖️  SMOTE เฉพาะ train  [อาจใช้เวลา 1-3 นาที]...")
    X_train, y_train = smote_with_progress(X_train, y_train)
    print(f"   → train หลัง SMOTE: {np.unique(y_train, return_counts=True)}")

    # ─── Per-tree progress bar ด้วย warm_start ───────────────────────────────
    print(f"🚀 เริ่มเทรน RandomForest ({N_ESTIMATORS} trees)...")

    clf = RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        max_depth=MAX_DEPTH,
        max_features=MAX_FEATURES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=42,
        n_jobs=N_JOBS,
    )

    BATCH = 50
    with tqdm(total=N_ESTIMATORS, desc="🌲 Training trees",
              unit="tree", ncols=80, colour="green") as pbar:
        while clf.n_estimators < N_ESTIMATORS:
            next_n = min(clf.n_estimators + BATCH, N_ESTIMATORS)
            clf.n_estimators = next_n
            clf.fit(X_train, y_train)
            pbar.update(BATCH)

    # ─── Evaluation ──────────────────────────────────────────────────────────
    print("\n🧠 กำลังประเมินโมเดล...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    np.save("rf_test_proba_clean.npy", y_proba)      # ใช้ใน ensemble_clean_new.py
    np.save("test_labels_clean.npy", y_test)
    from sklearn.metrics import roc_auc_score, f1_score
    print(f"Macro-AUC (รุ่นใหม่): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'):.4f}"
          f" | Macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy (RF): {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    importances = clf.feature_importances_
    top5_idx = np.argsort(importances)[::-1][:5]
    print("🔍 Top-5 feature indices:", top5_idx,
          "| importances:", np.round(importances[top5_idx], 4))

    joblib.dump(clf, MODEL_OUTPUT_PATH)
    print(f"💾 บันทึกโมเดล: {MODEL_OUTPUT_PATH}")

    # ─── Confusion Matrix ────────────────────────────────────────────────────
    print("\n📊 กำลังสร้าง Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix – Random Forest")
    plt.xlabel("Predicted labels"); plt.ylabel("True labels")
    plt.tight_layout()
    plt.savefig("rf_confusion_matrix_clean.png", dpi=150)
    plt.close()
    print("✅ บันทึก rf_confusion_matrix_clean.png")

    # ─── ROC Curve ───────────────────────────────────────────────────────────
    print("📈 กำลังสร้าง ROC Curve...")
    n_classes = len(CLASSES)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    y_score = clf.predict_proba(X_test)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(7, 6))
    colors = ['#e377c2', '#17becf', '#bcbd22']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f"{CLASSES[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot(all_fpr, mean_tpr, color='darkorange', lw=2.5, linestyle='--',
             label=f"Macro-avg (AUC = {macro_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Random Forest")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("rf_roc_curve_clean.png", dpi=150)
    plt.close()
    print("✅ บันทึก rf_roc_curve_clean.png")

if __name__ == "__main__":
    train_rf_model()