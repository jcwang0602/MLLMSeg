# MLLMSeg: Unlocking the Potential of MLLMs in Referring Expression Segmentation via a Light-weight Mask Decoder

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.37.2-green.svg)](https://huggingface.co/docs/transformers/)

<img src="assets/method.png" width="800">

## 📋 Overview

- [🚀 Quick Start](#-quick-start)
- [📊 Performance Metrics](#-performance-metrics)
- [👀 Visualization](#-visualization)
- [📦 Checkpoints](#-checkpoints)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

## 👀 Todo

- [ ] Release demo of MLLMSeg
- [ ] Add web-based demo interface

## 🚀 Quick Start

### Installation

```bash
conda create -n mllmseg python==3.9 -y
conda activate mllmseg
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Data Preparation
Referring segmentation datasets: [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip), [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip), [refCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) ([saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)) 

Generalized referring segmentation datasets: [gRefCOCO](https://github.com/henghuiding/gRefCOCO), we add the expressions and annotations json files into the `refer_seg` sub-directory, as shown in the tree structure below.


```angular2html
|-- datasets
│   ├── refer_seg
│   │   ├──grefcoco
│   │   ├── images
│   │   |   ├── saiapr_tc-12 
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
```

### Model Training

```bash
# Train RES
bash scripts/train_mllmseg_internvl.sh

# Train GRES
bash scripts/train_mllmseg_internvl_gres.sh
```

### Model Testing

```bash
# Test RES
bash scripts/test_mllmseg_internvl.sh

# Test GRES
bash scripts/test_mllmseg_internvl_gres.sh
```

### Merge Lora

```bash
python tools/merge_lora_mllmseg.py <input_path> <output_path>
```

## 📦 Checkpoints

Coming soon...

## 📊 Performance Metrics

### Referring Expression Segmentation
<img src="assets/tab_res.png" width="800">

### Referring Expression Comprehension
<img src="assets/tab_rec.png" width="800">

### Generalized Referring Expression Segmentation
<img src="assets/tab_gres.png" width="800">


## 👀 Visualization
### Referring Expression Segmentation
<img src="assets/res.png" width="800">

### Referring Expression Comprehension
<img src="assets/rec.png" width="800">

### Generalized Referring Expression Segmentation
<img src="assets/gres.png" width="800">

## 🙏 Acknowledgments
This code is developed on the top of [InternVL](https://github.com/OpenGVLab/InternVL), [GSVA](https://github.com/LeapLabTHU/GSVA), and [EEVG](https://github.com/chenwei746/EEVG).

## ✉️ Contact

Email: jcwang@stu.ecnu.edu.cn. Any kind discussions are welcomed!

---

## 📖 Citation
If our work is useful for your research, please consider cite:
```
@misc{wang2025progressivelanguageguidedvisuallearning,
      title={Progressive Language-guided Visual Learning for Multi-Task Visual Grounding}, 
      author={Jingchao Wang and Hong Wang and Wenlong Zhang and Kunhua Ji and Dingjiang Huang and Yefeng Zheng},
      year={2025},
      eprint={2504.16145},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.16145}, 
}
```