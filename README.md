# Repos for context-denoising-training

**Environmental Setup**:

We recommend using `transformers==4.46.1 to deploy models successfully.`

You may prepare the environment by running

```bash
conda create -n CDT python==3.10
conda activate CDT
pip install -r requirements.txt #(NOTE: 这个文件里需要补充，)

```

**Data Preparation:**

We use [pg19-test](https://huggingface.co/datasets/emozilla/pg19-test) dataset in our experiments. You may clone this repo by running

```bash
cd igscore/data
git clone https://huggingface.co/datasets/emozilla/pg19-test

```

## Preliminary

We generate datas from source data when testing. 

You may also get the full data :  **preliminary/data/full20.jsonl**

Recommentedly, get results with method where data generated online by running

```bash
python preliminary/src/test_score.py --model=meta-llama/Meta-Llama-3.1-8B-Instruct --context_lengths=7900
```

**[Note]  At least 8 GPUs with more than 85G memory of each are required to run successfully.**

Calcuate and visualize the IG / FR score of the model by running

```bash
python preliminary/src/stats_igscore.py --context_length=7900
python preliminary/src/stats_frscore.py --context_length=7900
```
