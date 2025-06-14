
# Repos for context-denoising-training

**Environmental Setup**

We recommend using ` transformers4.46.1` to deploy models successfully.

Install required packages by running

```bash
pip install -r requirements.txt
```

**Data Preparation**

We use [pg19-test](https://huggingface.co/datasets/emozilla/pg19-test) dataset in our experiments. You may clone this repo by running

```bash
cd preliminary/data
git clone https://huggingface.co/datasets/emozilla/pg19-test
```

## Preliminary

We generate data from source data when testing.

You may also use the full data, and we provide part of it:  **preliminary/data/full20.jsonl**

Our recommendation is to get results with method where data generated online by running

```bash
cd ../..
python preliminary/src/test_score.py --model=meta-llama/Meta-Llama-3.1-8B-Instruct --context_lengths=11900
```

**[Note]**

**At least 8 GPUs with more than 85G memory of each are required to run it successfully.**

Calculate and visualize the IG / FR score of the generated results by running

```bash
python preliminary/src/stats_igscore.py --context_length=11900
python preliminary/src/stats_frscore.py --context_length=11900
```
