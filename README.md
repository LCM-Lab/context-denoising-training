# Context Denoising Training for Long-Context Modeling

This is the official repository for the paper:  
**[Revisiting Long-Context Modeling from a Context Denoising Perspective](https://arxiv.org/abs/2510.05862)**.

---

## 🛠️ Environment Setup

We recommend using **`transformers==4.46.1`** to ensure successful model deployment.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📚 Data Preparation

We use the [**pg19-test**](https://huggingface.co/datasets/emozilla/pg19-test) dataset in our experiments. To download it, run:

```bash
cd preliminary/data
git clone https://huggingface.co/datasets/emozilla/pg19-test
```

### Preliminary Experiments

During evaluation, we dynamically generate test data from the source.  
A subset of the full evaluation data is provided at:  
`preliminary/data/full20.jsonl`

However, we **strongly recommend** generating data on-the-fly for the most accurate results. To do so, run:

```bash
cd ../..
python preliminary/src/test_score.py --model=meta-llama/Meta-Llama-3.1-8B-Instruct --context_lengths=11900
```

> **⚠️ Note**:  
> This step requires **at least 8 GPUs**, each with **more than 85 GB of memory**.

After generation, compute and visualize the **IG (Information Gain)** and **FR (Faithfulness Ratio)** scores:

```bash
python preliminary/src/stats_igscore.py --context_length=11900
python preliminary/src/stats_frscore.py --context_length=11900
```

---

## 🔥 Training

To set up the training environment, first clone the `LOOM-Train` framework:

```bash
git clone https://github.com/LCM-Lab/LOOM-Train.git
```

Then follow the setup instructions in the [LOOM-Train repository](https://github.com/LCM-Lab/LOOM-Train).

Once ready, launch the Context Denoising Training (CDT) process:

```bash
cd train
bash train_cdt.sh
```

---

## 🤝 Contributing
We welcome contributions! Whether it’s bug fixes, new features, or documentation improvements — feel free to open an issue or PR.

---

## 📬 Contact
Questions? Suggestions? Reach out at: zctang2000@gmail.com 

---

## 📚 Cite Us

If you find this work useful, please cite our paper:

```bibtex
@article{tang2025revisiting,
  title={Revisiting Long-context Modeling from Context Denoising Perspective},
  author={Tang, Zecheng and Ji, Baibei and Li, Juntao and Wu, Lijun and Gui, Haijia and Zhang, Min},
  journal={arXiv preprint arXiv:2510.05862},
  year={2025}
}
```





