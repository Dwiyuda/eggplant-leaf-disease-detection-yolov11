# Eggplant Leaf Disease Detection — YOLOv11

Object detection system for eggplant (*Solanum melongena*) leaf diseases. Instead of a single
label per photo, it returns **where** the symptoms are, as bounding boxes, across four classes:
`healthy_leaf`, `mosaic_virus_disease`, `leaf_spot_disease`, `insect_pest_disease`.

The public dataset was built for classification (one label per image). It was re-annotated
into a detection dataset: auto-labelling, a box-level audit with a prioritised review queue,
calibrated corrections, and re-training on the corrected labels.

## Results

Final model: **YOLO11s, image size 1024, 150 epochs**, trained on the corrected labels.

**Detection (test set)**

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.7224 | 0.6077 | 0.6736 | 0.5646 |

**Image-level diagnosis (test set, 208 leaves)** — detections are aggregated into one disease
call per leaf (confidence-sum rule, threshold chosen on the validation set by macro-F1):

| Accuracy | Macro-F1 | Correct |
|---:|---:|---:|
| **0.7981** | **0.8037** | 166 / 208 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| healthy_leaf | 0.889 | 0.774 | 0.828 |
| insect_pest_disease | 0.750 | 0.905 | 0.820 |
| mosaic_virus_disease | 0.800 | 0.821 | 0.810 |
| leaf_spot_disease | 0.815 | 0.707 | 0.757 |

## What is in the repo

| Folder | Content |
|---|---|
| `01_KODE/` | Dataset tooling: auto-labelling, label screening, box audit, calibration, split building, model comparison, camera-robustness and SAHI tests |
| `04_KAGGLE/` | Training scripts (splits 70:15:15 and 80:10:10, resolution ablation, P2 head, clean-variant comparison) and the final weights |
| `06_AUDIT_KOTAK/` | Box-audit outputs and calibration results |
| `07_SISTEM_WEB/` | Flask web demo: upload a leaf photo, get boxes, diagnosis, and disease information |

The images themselves (~9 GB) are not included.

## Run the web demo

```bash
git clone https://github.com/Dwiyuda/eggplant-leaf-disease-detection-yolov11.git
cd eggplant-leaf-disease-detection-yolov11
python -m venv .venv
.venv/Scripts/python -m pip install -r 07_SISTEM_WEB/requirements.txt   # macOS/Linux: .venv/bin/python
.venv/Scripts/python 07_SISTEM_WEB/app.py                               # http://localhost:5000
```

The final weights are already in the repo. More details in [07_SISTEM_WEB/README.md](07_SISTEM_WEB/README.md).

## Stack

Python · Ultralytics YOLOv11 · PyTorch · OpenCV · NumPy · Flask · Kaggle GPU

---

Built by **Dwi Yuda** · [dwiyuda.is-a.dev](https://dwiyuda.is-a.dev)
