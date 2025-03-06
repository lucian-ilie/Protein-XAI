import torch
from transformers import BertTokenizer, BertModel, T5EncoderModel, T5Tokenizer, logging
import re
import numpy as np
import sys
import ankh

logging.set_verbosity_error()

model_name = sys.argv[1]

if model_name == 'bert':
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
elif model_name == 't5':
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
elif model_name == 'ankh':
    model, tokenizer = ankh.load_base_model()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

for dataset in ['train', 'test', 'val']:
    with open(f'{dataset}.txt', 'r') as f:
        lines = f.readlines()

    PIDs = []
    Pseqs = []
    labels = []
    Pseq_lengths = []
    for i in range(0, len(lines), 3):
        PIDs.append(lines[i][1:-1])
        Pseqs.append(lines[i + 1][:-1])
        labels.append(lines[i + 2][:-1])
        Pseq_lengths.append(len(lines[i + 1][:-1]))

    if model_name != 'ankh':
        sequences = [' '.join(re.sub(r"[UZOB]", "X", sequence)) for sequence in Pseqs]
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, pad_to_max_length=False)
    else:
        sequences = [list(seq) for seq in Pseqs]
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, is_split_into_words=True)


    for i in range(len(PIDs)):
        input_ids = torch.tensor(ids['input_ids'][i]).to(device).reshape((1, -1))
        attention_mask = torch.tensor(ids['attention_mask'][i]).to(device).reshape((1, -1))
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        if model_name == 'bert':
            embedding = embedding.detach().cpu().numpy()[:, 1:-1, :]
        elif model_name == 't5' or model_name == 'ankh':
            embedding = embedding.detach().cpu().numpy()[:, :-1, :]
        with open(f'SeqInSite/{model_name} embeddings/{PIDs[i]}.npy', 'wb') as f:
            np.save(f, embedding)