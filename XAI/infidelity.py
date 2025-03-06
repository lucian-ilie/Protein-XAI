import torch
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer, logging
import ankh
import re
import captum
import numpy as np
import sys
import torch.nn as nn
from wrappers import BertWrapper, T5Wrapper
from captum.metrics import infidelity_perturb_func_decorator
import __main__


logging.set_verbosity_error()

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


class SeqInSite(nn.Module):
    
    def __init__(self, mlp_module, rnn_module):
        super(SeqInSite, self).__init__()
        self.mlp_module = mlp_module
        self.rnn_module = rnn_module

    def forward(self, embed, msa_embed):
        mlp_output = self.mlp_module(embed, msa_embed)
        rnn_output = self.rnn_module(embed, msa_embed)
        out = (mlp_output + rnn_output) / 2
        return out


@infidelity_perturb_func_decorator(multipy_by_inputs=False)
def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, 0.001, inputs.shape)).float().to(device)
    return inputs - noise

transformer = sys.argv[1]
xai_method = sys.argv[2]
protein_id = sys.argv[3]
sequence = sys.argv[4]

EMBEDDING_SIZE = 768 if transformer == 'ankh' else 1024
MSA_SIZE = 768
HALF_WINDOW = 4
num_aminoacids = len(sequence)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if transformer == 'bert':
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    embedding_model = BertModel.from_pretrained("Rostlab/prot_bert")
    embedding_layer = embedding_model.embeddings
    sequence= " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    wrapper = BertWrapper(embedding_model, xai_method)
elif transformer == 't5':
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    embedding_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    embedding_layer = embedding_model.encoder.embed_tokens
    sequence= " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    wrapper = T5Wrapper(embedding_model.encoder, xai_method)
elif transformer == 'ankh':
    embedding_model, tokenizer = ankh.load_base_model()
    embedding_layer = embedding_model.encoder.embed_tokens
    sequence= list(sequence)
    wrapper = T5Wrapper(embedding_model.encoder, xai_method)

embedding_model = embedding_model.to(device)
embedding_model = embedding_model.eval()

mlp_model = MLPModule(HALF_WINDOW, EMBEDDING_SIZE, MSA_SIZE)
mlp_model = torch.load(f'models/MLP_{transformer}_msa_torch.pt').to(device)
rnn_model = RNNModule(HALF_WINDOW, EMBEDDING_SIZE, MSA_SIZE)
rnn_model = torch.load(f'models/RNN_{transformer}_msa_torch.pt').to(device)
prediction_model = SeqInSite(mlp_model, rnn_model).to(device)
prediction_model = prediction_model.eval()

tokenizer_ids = tokenizer.batch_encode_plus([sequence], add_special_tokens=True, pad_to_max_length=False, is_split_into_words=True if transformer=='ankh' else False)
input_ids = torch.tensor([tokenizer_ids['input_ids'][0]]).to(device)    
inputs_embeds = embedding_layer(input_ids)

if transformer == 'bert':
    inputs_embeds = inputs_embeds[:, 1: -1, :]
elif transformer == 't5' or transformer == 'ankh':
    inputs_embeds = inputs_embeds[:, :-1, :]

emb_attr_address = f'{transformer}/embeddings interpretations/{protein_id}/{protein_id}_{xai_method}.npy'
pred_attr_address = f'{transformer}/SeqInSite interpretations/{protein_id}/{protein_id}_{xai_method}.npy'

with open(emb_attr_address, 'rb') as f:
    emb_attr = torch.tensor(np.load(f, allow_pickle=True))
if xai_method != 'GuidedGradCAM':
    with open(pred_attr_address, 'rb') as f:
        pred_attr = torch.tensor(np.load(f, allow_pickle=True))

    emb_infid = torch.zeros((num_aminoacids, EMBEDDING_SIZE))
    for k in range(num_aminoacids):
        for l in range(EMBEDDING_SIZE):
            emb_infid[k][l] = captum.metrics.infidelity(wrapper, perturb_fn, inputs_embeds, attributions=emb_attr[k][l].unsqueeze(0).to(device), additional_forward_args=(k, l, True))
    if xai_method != 'GuidedGradCAM':
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
        mlp_infid = captum.metrics.infidelity(prediction_model, perturb_fn, data, attributions=pred_attr.to(device), additional_forward_args=(msa,))

        pred_infid = emb_infid[4:-4, :].sum() * 9 + mlp_infid[4:-4].sum() * 9 
        + (emb_infid[0, :].sum() + emb_infid[-1, :].sum() + mlp_infid[0] + mlp_infid[-1]) * 5 
        + (emb_infid[1, :].sum() +  emb_infid[-2, :].sum() + mlp_infid[1] + mlp_infid[-2]) * 6 
        + (emb_infid[2, :].sum() +  emb_infid[-3, :].sum() + mlp_infid[2] + mlp_infid[-3]) * 7 
        + (emb_infid[3, :].sum() +  emb_infid[-4, :].sum() + mlp_infid[3] + mlp_infid[-4]) * 8

        pred_infid /= (max(0, num_aminoacids - 8) * 9 + 52)
        print('{} Prediction {}'.format(xai_method, pred_infid))
    
    print('{} Embedding {}'.format(xai_method, emb_infid.mean()))
    print()