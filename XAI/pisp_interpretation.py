import torch
import re
import numpy as np
import torch.nn as nn
import pandas as pd
from captum.attr import IntegratedGradients, DeepLift, GradientShap, InputXGradient, Saliency, GuidedBackprop, Deconvolution, Lime, KernelShap
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
import lzma
import pickle

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

class MLPModule(nn.Module):
    def __init__(self, half_window, embedding_size, msa_size):
        super(MLPModule, self).__init__()
        self.flatten = nn.Flatten()
        self.mlp_in = [(embedding_size + msa_size) * (2 * half_window + 1), 256, 128, 16]
        self.mlp_out = [256, 128, 16, 1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Linear(msa_size, msa_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        ])
        for i in range(len(self.mlp_in) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.mlp_in[i], self.mlp_out[i]),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
            )
        self.layers.append(
            nn.Sequential(
                nn.Linear(self.mlp_in[-1], self.mlp_out[-1]),
                nn.Sigmoid()
            )
        )
        

    def forward(self, embed, msa_embed):
        out1 = self.layers[0](embed)
        out2 = self.layers[1](msa_embed)
        out = torch.cat((out2, out1), dim=-1)
        out = self.flatten(out)
        for i in range(2, len(self.layers)):
            out = self.layers[i](out)
        return out
        

class RNNModule(nn.Module):
    def __init__(self, half_window, embedding_size, msa_size):
        super(RNNModule, self).__init__()
        self.mlp_in = [(embedding_size + msa_size), 256, 128, 16]
        self.mlp_out = [256, 128, 16, 1]
        self.flatten = nn.Flatten()
        self.rnn_embed = nn.ModuleList([
            nn.LSTM(input_size=embedding_size, hidden_size=64, num_layers=10, bidirectional=True, batch_first=True),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear((half_window * 2 + 1) * 64 * 2, embedding_size),
                nn.ReLU(),
                nn.Dropout(0.3)
        )])
        self.rnn_msa = nn.ModuleList([
            nn.LSTM(input_size=msa_size, hidden_size=64, num_layers=10, bidirectional=True, batch_first=True),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear((half_window * 2 + 1) * 64 * 2, msa_size),
                nn.ReLU(),
                nn.Dropout(0.3)
        )])
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.mlp_in[i], self.mlp_out[i]), 
                nn.ReLU(),
                nn.Dropout(0.3)
            ) 
            for i in range(3)])
        self.layers.append(
            nn.Sequential(
                nn.Linear(self.mlp_in[-1], self.mlp_out[-1]),
                nn.Sigmoid()
            )
        )
    
    def forward(self, embed, msa_embed):
        batch_size = embed.shape[0]
        h0 = torch.randn(2, 20, batch_size, 64).to(device)
        c0 = torch.randn(2, 20, batch_size, 64).to(device)
        embed_lstm_out, (hn, cn) = self.rnn_embed[0](embed, (h0[0], c0[0])) 
        embed_lstm_out = self.rnn_embed[1](embed_lstm_out)
        msa_lstm_out, (hn, cn) = self.rnn_msa[0](msa_embed, (h0[1], c0[1]))
        msa_lstm_out = self.rnn_msa[1](msa_lstm_out)
        out = torch.cat((msa_lstm_out, embed_lstm_out), dim=1)
        out = self.flatten(out)
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        return out


class SeqInSite_Wrapper(nn.Module):
    
    def __init__(self, mlp_module, rnn_module):
        super(SeqInSite_Wrapper, self).__init__()
        self.mlp_module = mlp_module
        self.rnn_module = rnn_module

    def forward(self, embed, msa_embed):
        mlp_output = self.mlp_module(embed, msa_embed)
        rnn_output = self.rnn_module(embed, msa_embed)
        out = (mlp_output + rnn_output) / 2
        return out

XAI_METHODS = {
    'Integrated Gradients': IntegratedGradients, 
    'DeepLift': DeepLift,
    'Gradient Shap': GradientShap, 
    'Input X Gradient': InputXGradient, 
    'Saliency': Saliency, 
    'Guided Backprop': GuidedBackprop, 
    'Deconvolution': Deconvolution, 
    'Lime': Lime, 
    'Kernel Shap': KernelShap, 
}

transformer = sys.argv[1]

EMBEDDING_SIZE = 768 if transformer == 'ankh' else 1024
MSA_SIZE = 768
HALF_WINDOW = 4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp_model = MLPModule(HALF_WINDOW, EMBEDDING_SIZE, MSA_SIZE)
mlp_model = torch.load(f'models/MLP_{transformer}_msa_torch.pt').to(device)
rnn_model = RNNModule(HALF_WINDOW, EMBEDDING_SIZE, MSA_SIZE)
rnn_model = torch.load(f'models/RNN_{transformer}_msa_torch.pt').to(device)
wrapper = SeqInSite_Wrapper(mlp_model, rnn_model).to(device)

with open('../human_short_seq.txt') as f:
    lines = f.readlines()

for i in range(0, len(lines) - 1, 3):
    protein_id = lines[i][1: -1]
    sequence = lines[i + 1][:-1]
    amino_acids = list(sequence)
    num_aminoacids = len(amino_acids)
    test_results = []
    size = 4 if len(amino_acids) < 40 else 3
    embedding = np.load(f'{transformer} embeddings/{protein_id}.npy')[0]
    embedding = np.vstack((np.zeros((HALF_WINDOW, EMBEDDING_SIZE)), embedding, np.zeros((HALF_WINDOW, EMBEDDING_SIZE))))
    msa_embedding = np.load(f'msa/{protein_id}.npy')
    msa_embedding = np.vstack((np.zeros((HALF_WINDOW, MSA_SIZE)), msa_embedding, np.zeros((HALF_WINDOW, MSA_SIZE))))
    data = np.zeros((num_aminoacids, 2 * HALF_WINDOW + 1, EMBEDDING_SIZE))
    msa = np.zeros((num_aminoacids, 2 * HALF_WINDOW + 1, MSA_SIZE))
    for i in range(num_aminoacids):
        data[i, :, :] = embedding[i:i + 2 * HALF_WINDOW + 1, :]
        msa[i, :, :] = msa_embedding[i:i + 2 * HALF_WINDOW + 1, :]
    data = torch.tensor(data).to(torch.float32).to(device)
    msa = torch.tensor(msa).to(torch.float32).to(device)
    for xai_method in XAI_METHODS.keys():
        print(protein_id, xai_method)
        emb_attr_address = '{}/embeddings interpretations/{}/{}_{}.xz'.format(transformer, protein_id, protein_id, xai_method.replace(' ', '').replace('-', ''))
        sis_attr_address = '{}/SeqInSite interpretations/{}/{}_{}.npy'.format(transformer, protein_id, protein_id, xai_method.replace(' ', '').replace('-', ''))
        pred_attr_address = '{}/predictions interpretations/{}/{}_{}.npy'.format(transformer, protein_id, protein_id, xai_method.replace(' ', '').replace('-', ''))
        figure_address = '{}/predictions interpretations figures/{}/{}_{}.jpg'.format(transformer, protein_id, protein_id, xai_method.replace(' ', '').replace('-', ''))
        with lzma.open(emb_attr_address, 'rb') as f:
            emb_attr = pickle.load(f)
        emb_attr = emb_attr.sum(axis=-1)
        First = True
        while First or np.isnan(psi_attr).any():
            First = False
            method = XAI_METHODS[xai_method](wrapper)
            if xai_method == 'Integrated Gradients':
                psi_attr = method.attribute(inputs=data, n_steps=20, additional_forward_args=(msa,)).detach().cpu().numpy()
            elif xai_method in ['DeepLift', 'Gradient Shap', 'Lime', 'Kernel Shap']:
                psi_attr = method.attribute(inputs=data, baselines=torch.zeros(data.shape).to(device), additional_forward_args=(msa,)).detach().cpu().numpy()
            else:
                psi_attr = method.attribute(inputs=data, additional_forward_args=(msa,)).detach().cpu().numpy()

        attr = np.zeros((num_aminoacids, num_aminoacids))
        for i in range(num_aminoacids):
            k_start = max(0, i - HALF_WINDOW)
            k_end = min(i + HALF_WINDOW + 1, num_aminoacids)
            j_start = max(HALF_WINDOW - i, 0)
            j_end = min(2 * HALF_WINDOW + 1, HALF_WINDOW + num_aminoacids - i)
            for j, k in zip(range(j_start, j_end), range(k_start, k_end)):
                attr[i] += np.matmul(psi_attr[i, j, :], emb_attr[k, :, :])

        with open(sis_attr_address, 'wb') as f:
            np.save(f, psi_attr)

        with open(pred_attr_address, 'wb') as f:
            np.save(f, attr)

        max_attr = np.max(np.abs(attr))
        attr /= max_attr
        df = pd.DataFrame(attr, index=amino_acids, columns=amino_acids)
        tr = 'Prot' if transformer != 'ankh' else ''
        plt.figure(figsize=(15, 12))
        sns.heatmap(df, linewidths=2, cmap='coolwarm', vmin=-1, vmax=1, annot=True, annot_kws={"fontsize":size}, fmt=".4f", square=True)
        plt.title('{}_{}_Predictions_{}_Chain {}'.format(xai_method, tr + transformer.title(), protein_id.upper()[:-1], protein_id.upper()[-1]))
        plt.savefig(figure_address, dpi=600, bbox_inches='tight')
        plt.close()
