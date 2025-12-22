
<div align="center">
<h1>SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery</h1>
<h3 align="center">NeurIPS 2025</h3>


<a href="https://openreview.net/pdf?id=B7lygdSDii" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2510.18740"><img src="https://img.shields.io/badge/arXiv-2503.18740-b31b1b" alt="arXiv"></a>
<a href="https://visual-ai.github.io/seal/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


**[Visual AI Lab, HKU](https://visailab.github.io/people.html)**

[Zhenqi He*](https://zhenqi-he.github.io/), [Yuanpei Liu*](https://scholar.google.com/citations?user=GHTB15QAAAAJ&hl=en), [Kai Han](https://www.kaihan.org/)
</div>

![teaser](assets/introduction.png)

## Prerequisite 🛠️

First, you need to clone the SEAL repository from GitHub. Open the terminal and run the following command:

```
git clone https://github.com/Visual-AI/SEAL.git
cd SEAL
```

We recommend setting up a conda environment for the project:

```
conda create --name=seal python=3.8
conda activate seal
pip install -r requirements.txt
```

Download the pretrained DINO/DINOv2 weights from their official repository to the ``PRETRAINED_PATH``.

## 📢 Updates
- [ ] 🛠️ **TODO:** We plan to release the trained model weights after the New Year Holiday.
- [2025/12/23] 🔥Released training and inference code for SSB Benchmarks.
- [2025/09/18] 🎉The paper was accepted by NeurIPS'25.

## Running 🏃
### Config

Set paths to datasets, pretrained weights, and log directories in ``config.py``.


### Datasets

We  use fine-grained benchmarks (CUB, Stanford-cars, FGVC-aircraft). You can find the datasets in:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb)


### Scripts
The scripts to train and eval each method can be found in the folder `/scripts`. 

**Eval the model**
```
bash scripts/eval.sh 
```

**Train the model**:

```
bash scripts/scars_dinov2.sh 
```

You may find the trained model weights in the following links: [here](https://drive.google.com/drive/folders/1K0rx6UebdUNfpFxm33CjQGEzwGit0eoD?usp=sharing). We plan to upload all model weights for both DINOv1 and DINOv2 shortly after the New Year holiday, due to ongoing checkpoint recovery and consolidation across multiple servers.


## Citing this work
<span id="jump"></span>
If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{He2025SEAL,
  author    = {Zhenqi He and Yuanpei Liu and Kai Han},
  title     = {SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  }

```