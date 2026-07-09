from pathlib import Path
import pandas as pd

# =====================================
# CONFIGURATION
# =====================================

LABELS_DIR = "/home/hiteshreddy/yolov5/runs/detect/exp/labels"

CLASS_NAMES = {
    0: "Bottle",
    1: "Cup"
}

# =====================================
# CHECK DIRECTORY
# =====================================

labels_path = Path(LABELS_DIR)

if not labels_path.exists():
    print(f"ERROR: Folder not found -> {LABELS_DIR}")
    exit()

# =====================================
# READ DETECTIONS
# =====================================

results = []

for txt_file in labels_path.glob("*.txt"):

    image_name = txt_file.stem

    with open(txt_file, "r") as f:

        for line in f:

            values = line.strip().split()

            if len(values) < 6:
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            confidence = float(values[5])

            results.append({
                "Image": image_name,
                "Class_ID": class_id,
                "Object": CLASS_NAMES.get(class_id, f"Class_{class_id}"),
                "Confidence": confidence,
                "X_center": x_center,
                "Y_center": y_center,
                "Width": width,
                "Height": height
            })

# =====================================
# CREATE DATAFRAME
# =====================================

df = pd.DataFrame(results)

if len(df) == 0:
    print("No detections found.")
    exit()

# =====================================
# DISPLAY ALL DETECTIONS
# =====================================

print("\n" + "=" * 60)
print("ALL DETECTIONS")
print("=" * 60)

print(df[["Image", "Object", "Confidence"]])

# =====================================
# STATISTICS
# =====================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total Detections : {len(df)}")

for obj in ["Bottle", "Cup"]:

    obj_df = df[df["Object"] == obj]

    if len(obj_df) == 0:
        continue

    print(f"\n{obj}")
    print("-" * 20)
    print(f"Count           : {len(obj_df)}")
    print(f"Average Conf    : {obj_df['Confidence'].mean():.4f}")
    print(f"Maximum Conf    : {obj_df['Confidence'].max():.4f}")
    print(f"Minimum Conf    : {obj_df['Confidence'].min():.4f}")

# =====================================
# TOP DETECTIONS
# =====================================

print("\n" + "=" * 60)
print("TOP 10 HIGHEST CONFIDENCE DETECTIONS")
print("=" * 60)

top10 = df.sort_values(
    by="Confidence",
    ascending=False
).head(10)

print(top10[["Image", "Object", "Confidence"]])

# =====================================
# SAVE CSV
# =====================================

csv_name = "confidence_report.csv"

df.to_csv(csv_name, index=False)

print("\nSaved CSV Report:")
print(csv_name)

# =====================================
# REPORT TABLE
# =====================================

print("\n" + "=" * 60)
print("REPORT TABLE")
print("=" * 60)

report = df[["Image", "Object", "Confidence"]]
print(report.to_string(index=False))





'''

============================================================
ALL DETECTIONS
============================================================
                                                Image  Object  Confidence
0   pexels-photo-12900865_webp.rf.bd899e0e6fef60f7...     Cup    0.409363
1   31Rsqgvl-tL_webp.rf.0642d6d57a77935d175434c7a8...     Cup    0.296197
2   31Rsqgvl-tL_webp.rf.0642d6d57a77935d175434c7a8...     Cup    0.641838
3   bottled-juice-on-supermarket-shelves_webp.rf.0...  Bottle    0.518092
4   plastic-bottles-for-recycle_webp.rf.de3bf52f63...  Bottle    0.406920
5   plastic-bottles-for-recycle_webp.rf.de3bf52f63...  Bottle    0.725730
6   plastic-bottles-for-recycle_webp.rf.de3bf52f63...  Bottle    0.821966
7   31Rsqgvl-tL_webp.rf.bddac8c3317d733891ca2a7178...     Cup    0.493592
8   WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp....     Cup    0.297785
9   WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp....     Cup    0.641845
10  WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp....     Cup    0.682508
11  HDCANPLAN1LTR_1_2d5968ec-ebcc-4965-8692-62aa0e...  Bottle    0.934738
12  WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp....     Cup    0.686593
13  WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp....     Cup    0.801087
14  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.722053
15  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.784606
16  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.805480
17  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.814195
18  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.872375
19  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.913485
20  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.456020
21  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.746662
22  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.789398
23  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.831845
24  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.872346
25  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.926886
26  plastic-bottles-for-recycle_webp.rf.a4ca28ae7e...  Bottle    0.351190
27  plastic-bottles-for-recycle_webp.rf.a4ca28ae7e...  Bottle    0.563376
28  plastic-bottles-for-recycle_webp.rf.a4ca28ae7e...  Bottle    0.650288
29  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.271824
30  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.290697
31  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.331828
32  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.478434
33  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.490323
34  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.524678
35  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.576936
36  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.584756
37  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.598403
38  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.680534
39  bottled-juice-on-supermarket-shelves_webp.rf.a...  Bottle    0.745704
40  HDCANPLAN1LTR_1_2d5968ec-ebcc-4965-8692-62aa0e...  Bottle    0.908591
41  ai-generated-8854229_640_webp.rf.1dbd91046fb7c...  Bottle    0.857014

============================================================
SUMMARY
============================================================
Total Detections : 42

Bottle
--------------------
Count           : 21
Average Conf    : 0.5863
Maximum Conf    : 0.9347
Minimum Conf    : 0.2718

Cup
--------------------
Count           : 21
Average Conf    : 0.6898
Maximum Conf    : 0.9269
Minimum Conf    : 0.2962

============================================================
TOP 10 HIGHEST CONFIDENCE DETECTIONS
============================================================
                                                Image  Object  Confidence
11  HDCANPLAN1LTR_1_2d5968ec-ebcc-4965-8692-62aa0e...  Bottle    0.934738
25  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.926886
19  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.913485
40  HDCANPLAN1LTR_1_2d5968ec-ebcc-4965-8692-62aa0e...  Bottle    0.908591
18  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.872375
24  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.872346
41  ai-generated-8854229_640_webp.rf.1dbd91046fb7c...  Bottle    0.857014
23  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.831845
6   plastic-bottles-for-recycle_webp.rf.de3bf52f63...  Bottle    0.821966
17  realistic-mug-mock-up-vector-template-easy-to-...     Cup    0.814195

Saved CSV Report:
confidence_report.csv

============================================================
REPORT TABLE
============================================================
                                                                                               Image Object  Confidence
                                      pexels-photo-12900865_webp.rf.bd899e0e6fef60f7fe30dc15edbf65fe    Cup    0.409363
                                                31Rsqgvl-tL_webp.rf.0642d6d57a77935d175434c7a857ccdf    Cup    0.296197
                                                31Rsqgvl-tL_webp.rf.0642d6d57a77935d175434c7a857ccdf    Cup    0.641838
                       bottled-juice-on-supermarket-shelves_webp.rf.0971fd5f5e2440c04cd3fb7d199175c0 Bottle    0.518092
                                plastic-bottles-for-recycle_webp.rf.de3bf52f63033f40397049f55a8681bd Bottle    0.406920
                                plastic-bottles-for-recycle_webp.rf.de3bf52f63033f40397049f55a8681bd Bottle    0.725730
                                plastic-bottles-for-recycle_webp.rf.de3bf52f63033f40397049f55a8681bd Bottle    0.821966
                                                31Rsqgvl-tL_webp.rf.bddac8c3317d733891ca2a71780da4e6    Cup    0.493592
                   WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp.rf.429b080df61d6901fd4a015836020dac    Cup    0.297785
                   WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp.rf.429b080df61d6901fd4a015836020dac    Cup    0.641845
                   WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp.rf.429b080df61d6901fd4a015836020dac    Cup    0.682508
 HDCANPLAN1LTR_1_2d5968ec-ebcc-4965-8692-62aa0e66bda5_1500x_webp.rf.2b7c52d7c4e8d67d54988205b9f4bc9f Bottle    0.934738
                   WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp.rf.15579eec352f4ffcc102a6321ecc9785    Cup    0.686593
                   WhatsApp_Image_2024-10-31_at_10-39-09_AM_webp.rf.15579eec352f4ffcc102a6321ecc9785    Cup    0.801087
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.cb4913755c59bb91fd08f9c509902851    Cup    0.722053
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.cb4913755c59bb91fd08f9c509902851    Cup    0.784606
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.cb4913755c59bb91fd08f9c509902851    Cup    0.805480
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.cb4913755c59bb91fd08f9c509902851    Cup    0.814195
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.cb4913755c59bb91fd08f9c509902851    Cup    0.872375
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.cb4913755c59bb91fd08f9c509902851    Cup    0.913485
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.a935cea3ccd34c0984d5525534f89d39    Cup    0.456020
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.a935cea3ccd34c0984d5525534f89d39    Cup    0.746662
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.a935cea3ccd34c0984d5525534f89d39    Cup    0.789398
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.a935cea3ccd34c0984d5525534f89d39    Cup    0.831845
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.a935cea3ccd34c0984d5525534f89d39    Cup    0.872346
realistic-mug-mock-up-vector-template-easy-to-change-colors_webp.rf.a935cea3ccd34c0984d5525534f89d39    Cup    0.926886
                                plastic-bottles-for-recycle_webp.rf.a4ca28ae7ee0471e52b8d3efa73ea694 Bottle    0.351190
                                plastic-bottles-for-recycle_webp.rf.a4ca28ae7ee0471e52b8d3efa73ea694 Bottle    0.563376
                                plastic-bottles-for-recycle_webp.rf.a4ca28ae7ee0471e52b8d3efa73ea694 Bottle    0.650288
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.271824
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.290697
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.331828
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.478434
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.490323
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.524678
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.576936
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.584756
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.598403
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.680534
                       bottled-juice-on-supermarket-shelves_webp.rf.a8b10efeb527f9d2271a971bbce4d39c Bottle    0.745704
 HDCANPLAN1LTR_1_2d5968ec-ebcc-4965-8692-62aa0e66bda5_1500x_webp.rf.54881c9207d9b3801a2508b37a14d9ac Bottle    0.908591
                                   ai-generated-8854229_640_webp.rf.1dbd91046fb7cbe2501a16963a2cb30e Bottle    0.857014
(yolov5env) hiteshreddy@REDDY:~/yolov5$ 



'''











