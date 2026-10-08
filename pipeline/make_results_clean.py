"""
make_results_clean.py — สรุปผลจริงจาก test set ที่ไม่ผ่านการ balance
=====================================================================
อ่านไฟล์ความน่าจะเป็นที่ train_*_clean_new.py เซฟไว้ แล้ว
  1) เขียน pipeline/RESULTS.md (ตาราง metric + sensitivity/specificity + confusion matrix)
  2) แทนที่ตารางใน ../README.md ระหว่าง <!-- RESULTS:START --> และ <!-- RESULTS:END -->
  3) เขียน portfolio_text_filled.txt (ข้อความพอร์ตที่ใส่ตัวเลขจริงแล้ว)

ใช้ได้ทั้งกรณีมีครบ RF+CNN และกรณีรัน RF อย่างเดียว (ยังไม่มีผล ensemble)
ตัวเลขทุกตัวคำนวณจากไฟล์ .npy ตรง ๆ ไม่มีการกรอกมือ
"""
import os
import sys
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "..", "README.md")
CLASSES = ["covid", "healthy", "symptomatic"]
ALPHA = 0.5   # น้ำหนัก CNN ใน soft voting (ตรงกับ app.py)
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def p(name):
    return os.path.join(HERE, name)


def load_optional(name):
    return np.load(p(name)) if os.path.exists(p(name)) else None


def metrics(y, proba):
    pred = proba.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class[c] = (tp / (tp + fn) if tp + fn else float("nan"),
                        tn / (tn + fp) if tn + fp else float("nan"))
    return {
        "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, average="macro", zero_division=0),
        "rec": recall_score(y, pred, average="macro", zero_division=0),
        "f1": f1_score(y, pred, average="macro", zero_division=0),
        "auc": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "cm": cm, "per_class": per_class,
    }


def pct(x):
    return f"{x * 100:.1f}"


def main():
    p_rf = load_optional("rf_test_proba_clean.npy")
    y_rf = load_optional("test_labels_clean.npy")
    p_cnn = load_optional("cnn_test_proba_clean.npy")
    y_cnn = load_optional("cnn_test_labels_clean.npy")
    if p_rf is None or y_rf is None:
        sys.exit("❌ ไม่พบผล RF — รัน train_rf_clean_new.py ก่อน")
    y = y_rf.astype(int)

    models = {"Random Forest": p_rf}
    if p_cnn is not None and y_cnn is not None:
        if len(y_cnn) != len(y) or not np.array_equal(y_cnn.astype(int), y):
            sys.exit("❌ test set ของ RF กับ CNN ไม่ตรงกัน — ห้ามรวมผล (ตรวจ manifest ของ CNN)")
        models["CNN"] = p_cnn
        models["CNN + RF (Ensemble)"] = ALPHA * p_cnn + (1 - ALPHA) * p_rf
    else:
        print("⚠️  ยังไม่มีผล CNN — สรุปเฉพาะ Random Forest")

    res = {k: metrics(y, v) for k, v in models.items()}
    n = len(y)
    counts = np.bincount(y, minlength=3)
    baseline = counts.max() / n

    # ---------- ตารางสำหรับ README ----------
    rows = ["| โมเดล | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Macro-AUC |",
            "|---|---|---|---|---|---|"]
    for k, m in res.items():
        rows.append(f"| {k} | {pct(m['acc'])}% | {pct(m['prec'])}% | {pct(m['rec'])}% | "
                    f"{pct(m['f1'])}% | {m['auc']:.2f} |")
    table = "\n".join(rows)
    note = (f"\nชุดทดสอบ {n:,} ไฟล์ (covid {counts[0]:,} / healthy {counts[1]:,} / symptomatic {counts[2]:,}) "
            f"คงสัดส่วนจริงของข้อมูล ไม่ผ่านการ balance · ถ้าทายว่า Healthy ทุกไฟล์จะได้ Accuracy ≈ {pct(baseline)}% "
            f"จึงใช้ **Macro-F1** และ **Sensitivity รายคลาส** เป็นตัวชี้วัดหลัก")
    block = f"{START}\n{table}\n{note}\n{END}"

    if os.path.exists(README):
        s = open(README, encoding="utf-8").read()
        if START in s and END in s:
            a, b = s.index(START), s.index(END) + len(END)
            open(README, "w", encoding="utf-8", newline="\n").write(s[:a] + block + s[b:])
            print("✅ อัปเดตตารางใน README.md")
        else:
            anchor = "## ข้อจำกัด"
            section = "### ผลบนชุดทดสอบสัดส่วนจริง (แยกชุดทดสอบก่อนปรับสมดุล)\n\n" + block + "\n\n"
            if anchor in s:
                i = s.index(anchor)
                open(README, "w", encoding="utf-8", newline="\n").write(s[:i] + section + s[i:])
                print("✅ เพิ่มหัวข้อผลบนชุดทดสอบสัดส่วนจริงใน README.md")
            else:
                print("⚠️  README ไม่มีหัวข้อ '## ข้อจำกัด' — ไม่ได้แทรกตาราง")

    # ---------- RESULTS.md ----------
    out = ["# ผลการประเมิน (test set จริง ไม่ผ่านการ balance)\n", table, note, ""]
    for k, m in res.items():
        out.append(f"\n## {k}\n")
        out.append("| คลาส | Sensitivity | Specificity |\n|---|---|---|")
        for c, (se, sp) in m["per_class"].items():
            out.append(f"| {c} | {se:.3f} | {sp:.3f} |")
        cm = m["cm"]
        out.append("\nConfusion matrix (แถว = จริง, คอลัมน์ = ทำนาย; covid / healthy / symptomatic)\n")
        out.append("```")
        for i, c in enumerate(CLASSES):
            out.append(f"{c:<12}" + "".join(f"{v:>7}" for v in cm[i]))
        out.append("```")
    open(p("RESULTS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print("✅ เขียน pipeline/RESULTS.md")

    # ---------- ข้อความพอร์ต ----------
    if "CNN + RF (Ensemble)" in res:
        e = res["CNN + RF (Ensemble)"]
        se_covid = e["per_class"]["covid"][0]
        text = (
            "ประชากรโลกกว่า 4,600 ล้านคนยังเข้าไม่ถึงบริการสุขภาพพื้นฐานที่จำเป็น และข้อจำกัดด้านระยะทาง เวลา และค่าใช้จ่ายทำให้"
            "ผู้ที่มีอาการไอและมีความเสี่ยงต่อโรคทางเดินหายใจเข้าถึงการประเมินเบื้องต้นได้ล่าช้า Cough AI จึงพัฒนาเป็นเว็บแอปพลิเคชัน"
            "ที่ใช้งานผ่านเบราว์เซอร์บนสมาร์ตโฟนได้ทันที โดยไม่ต้องติดตั้งแอปหรือใช้อุปกรณ์เพิ่มเติม ผู้ใช้บันทึกหรืออัปโหลดเสียงไอ"
            "ประมาณ 5 วินาที ระบบวิเคราะห์และแสดงความน่าจะเป็นของ 3 กลุ่ม ได้แก่ Healthy, COVID-19 และ Symptomatic พร้อมระดับความเสี่ยง"
            "ภายในไม่กี่วินาที โดยใช้ Ensemble แบบ Soft-Voting ระหว่าง CNN ที่เรียนรู้จาก Mel-Spectrogram และ Random Forest ที่ใช้ "
            "Audio Features จำนวน 416 มิติ ฝึกด้วยชุดข้อมูลเสียงไอสาธารณะ COUGHVID จำนวน 20,188 ไฟล์ "
            "(Healthy 15,091, Symptomatic 3,807, COVID-19 1,290) "
            f"ผลทดสอบบนชุดทดสอบ {n:,} ไฟล์ที่แยกไว้ก่อนการปรับสมดุลข้อมูล และคงสัดส่วนจริงของข้อมูล Ensemble ได้ "
            f"Macro F1 {pct(e['f1'])}%, Macro Precision {pct(e['prec'])}%, Macro Recall {pct(e['rec'])}%, "
            f"Accuracy {pct(e['acc'])}% และ Macro AUC {e['auc']:.2f} โดยตรวจจับกลุ่ม COVID-19 ได้ (Sensitivity) {pct(se_covid)}% "
            "ทั้งนี้เป็นเครื่องมือคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยทางการแพทย์ "
            "ข้าพเจ้ารับหน้าที่ Team Leader และ Developer โดยรับผิดชอบกระบวนการพัฒนาตั้งแต่ต้นจนจบ ตั้งแต่การเตรียมและประมวลผลข้อมูลเสียงไอ "
            "การทำ Feature Extraction โดยสกัดคุณลักษณะจากเสียง เช่น MFCC, Mel-Spectrogram และคุณลักษณะด้าน Spectral และ Temporal รวม 416 มิติ "
            "เพื่อนำไปใช้กับโมเดล Random Forest ขณะเดียวกันได้ออกแบบกระบวนการแปลงเสียงเป็น Mel-Spectrogram สำหรับให้ CNN เรียนรู้รูปแบบของเสียง "
            "จากนั้นพัฒนา Ensemble แบบ Soft-Voting เพื่อรวมผลจาก CNN และ Random Forest และพัฒนา Flask API เชื่อมต่อโมเดลกับเว็บแอปพลิเคชัน "
            "รวมถึงทดสอบและปรับปรุงระบบให้สามารถวิเคราะห์เสียงไอและแสดงผลการคัดกรองแก่ผู้ใช้ได้\n\n"
            "Website : https://cough-ai-wine.vercel.app\n"
            "Source code : https://github.com/lilpaint1/Cough_AI\n"
            "Video : https://youtu.be/FlX0uUY_zhg\n"
        )
        open(p("portfolio_text_filled.txt"), "w", encoding="utf-8", newline="\n").write(text)
        print("✅ เขียน pipeline/portfolio_text_filled.txt (ข้อความพอร์ตที่ใส่ตัวเลขจริงแล้ว)")
    else:
        print("ℹ️  ข้อความพอร์ตต้องใช้ผล Ensemble — รัน CNN แล้วรันสคริปต์นี้อีกครั้ง")

    print("\n=== สรุป ===")
    print(table)
    print(note.strip())


if __name__ == "__main__":
    main()
