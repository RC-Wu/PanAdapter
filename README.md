# PanAdapter (AAAI 2025 Oral) — PyTorch Implementation

This repository provides a PyTorch implementation of **PanAdapter** for **remote sensing pansharpening** (LRMS + PAN → HRMS), following the paper:

**PanAdapter: Two-Stage Fine-Tuning with Spatial-Spectral Priors Injecting for Pansharpening**  
[[AAAI Page]](https://ojs.aaai.org/index.php/AAAI/article/view/32912) [[arXiv]](https://arxiv.org/abs/2409.06980) 

Our codebase is built **on top of** (and modified from) the UDL framework:  
[[UDL (XiaoXiao-Woo)]](https://github.com/XiaoXiao-Woo/UDL)

> **Two-stage training overview (as in the paper):**
> - **Stage 1**: Fine-tune task-specific *Local Prior Extraction (LPE)* modules to extract two-scale priors (spectral + spatial).
> - **Stage 2**: Feed priors into cascaded adapters, injecting them into a **frozen** pre-trained **IPT** (VisionTransformer) backbone.

---

## Pretrained Weights

This project relies on **two pretrained models** (as in the paper): **IPT** and **EDSR**. Please download and place them in the expected paths.

### 1) IPT pretrained weights

Download `IPT_pretrain.pt` from the following Google Drive folder:  
https://drive.google.com/drive/folders/1MVSdUX0YBExauG0fFz4ANiWTrq9xZEj7

**Suggested location**
```text
PreWeight/IPT/IPT_pretrain.pt
```

### 2) EDSR pretrained weights

Download:
https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt

**Suggested location**
```text
PreWeight/EDSR/edsr_baseline_x2-1bc95232.pt
```

---

## Environment Setup

Our implementation is based on the UDL framework. Please follow UDL’s basic setup guidelines first:  
[[UDL GitHub]](https://github.com/XiaoXiao-Woo/UDL)

---

## Dataset Configuration

Training uses `.h5` datasets. The training entry script is:

```text
baseline/train.py
```

Before training, **edit the dataset paths in `baseline/train.py`**:

- **Line 39**: `train_data_path = ...`
- **Line 40**: `test_data_path = ...`

Set them to your local paths (examples):
```python
train_data_path = "/path/to/train_wv3.h5"
test_data_path  = "/path/to/test_wv3.h5"
```

---

## Training (Stage 1 / Stage 2)

Training is performed via:
```bash
python baseline/train.py
```

### Stage 1: Train SSPEN / LPE (priors extraction)

In `baseline/train.py`, set the model to **stage 1** before training:
```python
model.set_stage(1)
```

Then run:
```bash
python baseline/train.py
```

### Stage 2: Train PanAdapter (adapters + INR tail) — main result

In `baseline/train.py`, switch to **stage 2**:
```python
model.set_stage(2)
```

Then run:
```bash
python baseline/train.py
```


## Acknowledgements

This project is built upon and inspired by several excellent works and open-source efforts:

- **UDL framework** (base training/infrastructure):  
  [[UDL]](https://github.com/XiaoXiao-Woo/UDL)

- **IPT (Pre-Trained Image Processing Transformer)**:  
  [[CVPR 2021 PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.pdf)  
  [[arXiv]](https://arxiv.org/abs/2012.00364)

- **EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution)**:  
  [[Official PyTorch Repo]](https://github.com/sanghyun-son/EDSR-PyTorch)  
  [[arXiv]](https://arxiv.org/abs/1707.02921)

Special thanks to the authors and maintainers for their valuable contributions to the community.