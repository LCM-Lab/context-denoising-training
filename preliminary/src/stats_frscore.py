import argparse, os
import pickle

    

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import FancyBboxPatch


def normalize(datas):
    s = sum(datas)
    return [k/s for k in datas]

def draw(datas, path):

    ret = []

    for score in [100 , 0]:
        samples_number = 0
        for data in datas:
            if data['score']!=score:continue
            samples_number += 1
            supporting_pos = data['evidence_pos']
            interference_pos = data['attack_pos']
            lowfrequence_pos = data['emoji_pos']

            supporting_length = sum(x[1]-x[0] for x in supporting_pos)
            interference_length = sum(x[1]-x[0] for x in interference_pos)
            irrelevant_length = data['irr_length'] 
            lowfrequence_length = sum(x[1]-x[0] for x in lowfrequence_pos)
            scores = [0,0,0,0]
            lengths = [supporting_length, interference_length, irrelevant_length, lowfrequence_length]

            for l in range(24, 32):
                for i in range(len(scores)):
                    scores[i] += data['attention'][l]['weight']['score'][i] * lengths[i]
            
            for i in range(len(scores)):
                scores[i]/=8
        
        for i in range(len(scores)):
            scores[i]/=max(1,samples_number)

        ret += [scores]

    data = pd.DataFrame({
        'Context Type': ['Supporting', 'Interference', 'Irrelevant', 'Low-frequency'],
        'correct': normalize(ret[0]),
        'wrong': normalize(ret[1]),
    })



    data_melted = data.melt(id_vars='Context Type', var_name='Predicted Result', value_name='Mean Weight')

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Context Type', y='Mean Weight', hue='Predicted Result', data=data_melted, width = 0.6)


    plt.title('FR score', fontsize=28)
    plt.ylabel('score value', fontsize=28)



    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.1f').rstrip('0').rstrip('.').lstrip("0"), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True)) 
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))


    plt.legend(title = '', loc = 'upper left', prop={'size':28})
    plt.tight_layout()

    print("save:",path)
    plt.savefig(path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--context_length', type=int, default=7900)
    parser.add_argument('--result_dir', type=str, default='information_flow')
    parser.add_argument('--model_tag',type = str, default='Meta-Llama-3.1-8B-Instruct')

    args = parser.parse_args()
    results_dir = f"preliminary/results/{args.result_dir}/{args.model_tag}/{args.context_length}/label/"

    file_paths = [os.path.join(results_dir,k) for k in os.listdir(results_dir)]

    file_datas = [pickle.load(open(k,'rb')) for k in file_paths]

    draw(file_datas, f"preliminary/results/figures/FR-{args.result_dir}-{args.model_tag}-{args.context_length}.pdf")



