import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import intervaltree
import torchmetrics
import pandas as pd
import sys
import math
import warnings
from sklearn.metrics import roc_curve, auc, precision_recall_curve

warnings.filterwarnings('ignore')

class ProteinSet(Dataset):
    def __init__(self, interval_tree, half_window, batches, labels, embeddings, msa):
        self.interval_tree = interval_tree
        self.half_window = half_window
        self.batches = batches
        self.labels = labels
        self.embeddings = embeddings
        self.msa = msa

    def __len__(self):
        return len(self.batches)

    def __make_batch(self, batch_id):
        batch_embeds = []
        batch_msa = []
        batch_labels = []
        for index in self.batches[batch_id]:
            protein_data = self.interval_tree.at(index).pop()
            aa_index = index - protein_data[0]
            pid = protein_data[2]
            batch_embeds.append(self.embeddings[pid][aa_index:aa_index + 2 * self.half_window + 1])
            batch_msa.append(self.msa[pid][aa_index:aa_index + 2 * self.half_window + 1])
            batch_labels.append([float(self.labels[pid][aa_index])])
        return (batch_embeds, batch_msa, batch_labels)

    def __getitem__(self, index):
        batch = self.__make_batch(index)
        embed = torch.tensor(batch[0], dtype=torch.float32)
        msa = torch.tensor(batch[1], dtype=torch.float32)
        label = torch.tensor(batch[2])
        return [embed, msa, label]


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


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model, type, transformer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, amsgrad=AMSGRAD)
    accuracy = torchmetrics.classification.BinaryAccuracy().to(device)
    early_stopper = EarlyStopper(patience=PATIENCE)
    train_log = pd.DataFrame({'loss': np.zeros((EPOCHS,)), 'accuracy': np.zeros((EPOCHS,))})
    val_log = pd.DataFrame({'loss': np.zeros((EPOCHS,)), 'accuracy': np.zeros((EPOCHS,))})
    min_val_loss = np.inf
    
    for epoch in range(EPOCHS):
        print('-----------Epoch {}-----------'.format(epoch))
    
        model.train()
        for _, (embed, msa_embed, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(embed[0].to(device), msa_embed[0].to(device))
            batch_loss = loss(outputs, targets[0].to(device))
            batch_loss.backward()
            optimizer.step()
            train_log['loss'][epoch] += batch_loss.item() * embed.shape[1]
            train_log['accuracy'][epoch] += accuracy(outputs, targets[0].to(device)).detach().cpu().numpy() * embed.shape[1]
    
        train_log.loc[epoch] = train_log.loc[epoch] / total_length['train']
        logs = tuple(train_log.loc[[epoch]].values[0])
        print('--Train-- {} Loss: {:.5f} {} Accuracy: {:.5f}'.format(type, logs[0], type, logs[1]))
    
        model.eval()
        for _, (embed, msa_embed, targets) in enumerate(val_loader):
            outputs = model(embed[0].to(device), msa_embed[0].to(device))
            batch_loss = loss(outputs, targets[0].to(device))
            val_log['loss'][epoch] += batch_loss.item() * embed.shape[1]
            val_log['accuracy'][epoch] += accuracy(outputs, targets[0].to(device)).detach().cpu().numpy() * embed.shape[1]
            
        val_log.loc[epoch] = val_log.loc[epoch] / total_length['val']
        logs = tuple(val_log.loc[[epoch]].values[0])
        print('--Validation-- {} Loss: {:.5f} {} Accuracy: {:.5f}'.format(type, logs[0], type, logs[1]))
    
        if val_log['loss'][epoch] < min_val_loss:
            torch.save(model, 'models/{}_{}_msa_torch.pt'.format(type, transformer))
            min_val_loss = val_log['loss'][epoch]
            print('Saved best model!')
            
        if early_stopper.early_stop(val_log['loss'][epoch]):             
            break


def calculate_evaluation_metrics(y_true, predictions):
    fpr, tpr, thresholds = roc_curve(y_true, predictions)
    au_roc = auc(fpr, tpr)
    print("Area under ROC curve: ", au_roc)
    precision, recall, thresholds = precision_recall_curve(y_true, predictions)
    aupr = auc(recall, precision)
    print("Area under PR curve: ", aupr)
    sorted_pred = np.sort(predictions)
    sorted_pred_descending = np.flip(sorted_pred)
    num_of_1 = np.count_nonzero(y_true)
    threshold = sorted_pred_descending.item(num_of_1 - 1)
    print('num_of_1: ' + str(num_of_1))
    print("threshold: " + str(threshold))
    pred_binary_sum = sum(list(np.where(predictions > threshold, 1, 0)))
    y_pred = []
    flag = 0
    for item in predictions:
        if item == threshold:
            if flag < num_of_1 - pred_binary_sum:
                y_pred.append(1)
                flag += 1
            else:
                y_pred.append(0)
        elif item > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    TP = float(0)
    FP = float(0)
    TN = float(0)
    FN = float(0)
    for i, j in zip(y_true, y_pred):
        if (i == 1 and j == 1):
            TP += 1
        elif (i == 0 and j == 1):
            FP += 1
        elif (i == 0 and j == 0):
            TN += 1
        elif (i == 1 and j == 0):
            FN += 1
    print("TP: ", TP)
    print("FP: ", FP)
    print("TN: ", TN)
    print("FN: ", FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Sensitivity: " + str(sensitivity))
    print("Specificity: " + str(specificity))
    print("Recall: " + str(recall))
    print("Precision: " + str(precision))
    print("MCC: " + str(MCC))
    print("F1: " + str(F1))
    print("Accuracy: " + str(accuracy))

transformer = sys.argv[1]

HALF_WINDOW = 4
EMBEDDING_SIZE = 768 if transformer == 'ankh' else 1024
MSA_SIZE = 768
BATCH_SIZE = 1024
LR = 0.001
PATIENCE = 4
EPOCHS = 100
BETAS = (0.9, 0.999)
AMSGRAD = False
NUM_WORKERS = 16


labels = {'train': {}, 'val': {}, 'test': {}}
total_length = {'train': 0, 'val': 0, 'test': 0}
tree = {'train': None, 'val': None, 'test': None}
embeddings = {'train': {}, 'val': {}, 'test': {}}
msa = {'train': {}, 'val': {}, 'test': {}}
batches = {'train': [], 'val': [], 'test': []}
for s in ['train', 'val', 'test']:
    with open(('{}.txt'.format(s)), 'r') as f:
        lines = f.readlines()
    tree[s] = intervaltree.IntervalTree()
    batches[s].append([])
    for i in range(0, len(lines), 3):
        prot_id = lines[i][1:-1]
        prot_length = len(lines[i + 1][:-1])
        labels[s][prot_id] = lines[i + 2][:-1]
        prot_embedding = np.load(f'{transformer} embeddings/{prot_id}.npy')[0]
        prot_embedding = np.vstack((np.zeros((HALF_WINDOW, EMBEDDING_SIZE)), prot_embedding, np.zeros((HALF_WINDOW, EMBEDDING_SIZE))))
        embeddings[s][prot_id] = prot_embedding
        prot_msa = np.load(f'msa/{prot_id}.npy')
        prot_msa = np.vstack((np.zeros((HALF_WINDOW, MSA_SIZE)), prot_msa, np.zeros((HALF_WINDOW, MSA_SIZE))))
        msa[s][prot_id] = prot_msa
        tree[s].addi(total_length[s], total_length[s] + prot_length, prot_id)
        if len(batches[s][-1]) < BATCH_SIZE:
            batches[s][-1] = batches[s][-1] + list(range(total_length[s], total_length[s] + prot_length))
        else:
            batches[s].append(list(range(total_length[s], total_length[s] + prot_length)))
        total_length[s] += prot_length
            

train_set = ProteinSet(tree['train'], HALF_WINDOW, batches['train'], labels['train'], embeddings['train'], msa['train'])
val_set = ProteinSet(tree['val'], HALF_WINDOW, batches['val'], labels['val'], embeddings['val'], msa['val'])
test_set = ProteinSet(tree['test'], HALF_WINDOW, batches['test'], labels['test'], embeddings['test'], msa['test'])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

my_mlp_model = MLPModule(HALF_WINDOW, EMBEDDING_SIZE, MSA_SIZE)
print('Total Number of Parameters: ' + str(sum(p.numel() for p in my_mlp_model.parameters())))
print('Number of Trainable Parameters: ' + str(sum(p.numel() for p in my_mlp_model.parameters() if p.requires_grad)))
print('Number of Non-trainable Parameters: ' + str(sum(p.numel() for p in my_mlp_model.parameters() if not p.requires_grad)))
print(my_mlp_model)
train(my_mlp_model, 'MLP', transformer)

my_rnn_model = MLPModule(HALF_WINDOW, EMBEDDING_SIZE, MSA_SIZE)
print('Total Number of Parameters: ' + str(sum(p.numel() for p in my_rnn_model.parameters())))
print('Number of Trainable Parameters: ' + str(sum(p.numel() for p in my_rnn_model.parameters() if p.requires_grad)))
print('Number of Non-trainable Parameters: ' + str(sum(p.numel() for p in my_rnn_model.parameters() if not p.requires_grad)))
print(my_rnn_model)
train(my_rnn_model, 'RNN', transformer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_mlp_model = torch.load('models/MLP_{}_msa_torch.pt'.format(transformer)).to(device)
my_rnn_model = torch.load('models/RNN_{}_msa_torch.pt'.format(transformer)).to(device)
predictions = []
test_labels = []
with torch.no_grad():
    for _, (embed, msa_embed, targets) in enumerate(test_loader):
        embed, msa_embed, targets = embed.to(device), msa_embed.to(device), targets.to(device)
        mlp_outputs = my_mlp_model(embed[0], msa_embed[0])
        rnn_outputs = my_rnn_model(embed[0], msa_embed[0])
        predictions.append((mlp_outputs.detach().cpu().numpy() + rnn_outputs.detach().cpu().numpy()) / 2)
        test_labels.append(targets[0].detach().cpu().numpy())
predictions = np.concatenate(predictions).reshape((-1,))
test_labels = np.concatenate(test_labels).reshape((-1,))
calculate_evaluation_metrics(test_labels, predictions)
