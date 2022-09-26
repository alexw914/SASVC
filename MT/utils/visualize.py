import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def reduce_dimension(target_dict, gt_data, output_name, hparams):

    X = list(target_dict.values())
    X = TSNE(n_components=2,perplexity=40, init='random').fit_transform(X)
    utts = list(target_dict.keys())
    arr = []
    for i in range(len(utts)):
        utt_id = utts[i]
        if gt_data[utt_id]['cm_id'] == 'bonafide':
            arr.append(np.append(X[i], [gt_data[utt_id]['speaker_id'], utt_id]))
        else:
            arr.append(np.append(X[i], ['spoof', utt_id] ))
    df = pd.DataFrame(arr, columns=['x','y','label','utt_id'])
    df.to_csv(os.path.join(hparams['output_folder'], output_name))


def visualize(df=None, spk=None, show="spk"):

    spoof_proto_file = 'cm/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    with open(spoof_proto_file, 'r') as f:
        spoof_proto = f.readlines()
    spoof_id_dict = {}
    for line in spoof_proto:
        line = line.split()
        spoof_id_dict[line[1]] = line[3]

    data = []
    if show == "cm":
        for index, row in df.iterrows():
            cur = []
            cur.append(row['x'])
            cur.append(row['y'])
            if row['label'] != 'spoof':
                cur.append('bonafide')
            else:
                spoof_id = spoof_id_dict[row['utt_id']]
                if spoof_id in ['A05','A06','A17','A18','A19']:
                    cur.append('vc')
                else:
                    cur.append('tts')
            data.append(cur)

    if show=="spk":
        for index, row in spk.iterrows():
            cur = []
            cur.append(row['x'])
            cur.append(row['y'])
            cur.append(row["label"])
            data.append(cur)
        
    data = pd.DataFrame(data, columns=['x','y','label'])
    if show=="spk":
        ax =sns.scatterplot(x='x', y='y', data=data, hue='label', s=8, legend=None)
    else:
        ax =sns.scatterplot(x='x', y='y', data=data, hue='label', s=3)

    ax.set(xlabel=None)
    ax.set(ylabel=None)
    plt.show()
    plt.close()

if __name__=="__main__":
    df = pd.read_csv('results/sasv/1678/all_2d.csv')
    spk = pd.read_csv('results/sasv/1678/spk_2d.csv')
    visualize(df, spk, show="spk")