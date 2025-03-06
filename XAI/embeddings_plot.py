import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lzma
import pickle


XAI_METHODS = [
    'Integrated Gradients',
    'DeepLift',
    'Gradient Shap',
    'Input X Gradient', 
    'Saliency', 
    'Guided Backprop', 
    'Deconvolution', 
    'Lime', 
    'Kernel Shap', 
]

with open('../human_short_seq.txt') as f:
    lines = f.readlines()

for i in range(0, len(lines) - 1, 3):
    protein_id = lines[i][1: -1]
    sequence = lines[i + 1][:-1]
    interactions = lines[i + 2][:-1]
    amino_acids = list(sequence)
    size = 4 if len(amino_acids) < 40 else 3.5
    for transformer in ['ankh', 'bert', 't5']:
        for xai_method in XAI_METHODS:
            emb_attr_address = '{}/embeddings interpretations/{}/{}_{}.xz'.format(transformer, protein_id, protein_id, xai_method.replace(' ', '').replace('-', ''))
            figure_address = '{}/embeddings interpretations figures/{}/{}_{}.jpg'.format(transformer, protein_id, protein_id, xai_method.replace(' ', '').replace('-', ''))
            try:
                with lzma.open(emb_attr_address, 'rb') as f:
                    attr = pickle.load(f)
            except:
                print('{} {} {}'.format(protein_id, transformer, xai_method))
                break
            if len(attr.shape) != 4:
                print('{} {} {}'.format(protein_id, transformer, xai_method))
                break
            attr = attr.sum(axis=-1).sum(axis=1)
            max_attr = np.max(np.abs(attr))
            attr /= max_attr
            df = pd.DataFrame(attr, index=amino_acids, columns=amino_acids)
            plt.figure(figsize=(15, 12))
            ax = sns.heatmap(df, linewidths=2, cmap=sns.diverging_palette(12, 120, s=100,l=35, as_cmap=True), vmin=-1, vmax=1, annot=True, annot_kws={"fontsize":size}, fmt=".4f", square=True)
            xticks = ax.get_xticklabels()
            yticks = ax.get_yticklabels()
            for i in range(len(amino_acids)):
                if int(interactions[i]) == 1:
                    xticks[i].set_color(sns.set_hls_values('g', h=0.3, l=0.35, s=1))
                    yticks[i].set_color(sns.set_hls_values('g', h=0.3, l=0.35, s=1))
                else:
                    xticks[i].set_color(sns.set_hls_values('r', h=0.028, l=0.35, s=1))
                    yticks[i].set_color(sns.set_hls_values('r', h=0.028, l=0.35, s=1))
            plt.title('{}_{}_Embeddings_{}_Chain {}'.format(xai_method, transformer.title(), protein_id.upper()[:-1], protein_id.upper()[-1]))
            plt.savefig(figure_address, dpi=600, bbox_inches='tight')
            plt.close()