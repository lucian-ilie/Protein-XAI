import torch
from wrappers import BertWrapper, T5Wrapper
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer, logging
import ankh
from captum.attr import IntegratedGradients, DeepLift, GradientShap, InputXGradient, Saliency, GuidedBackprop, Deconvolution, GuidedGradCam, Lime, KernelShap
import re
import numpy as np
import sys
import time
import pickle
import lzma

logging.set_verbosity_error()

XAI_METHODS = {
    'IntegratedGradients': IntegratedGradients, 
    'DeepLift': DeepLift,
    'GradientShap': GradientShap, 
    'InputXGradient': InputXGradient, 
    'Saliency': Saliency, 
    'GuidedBackprop': GuidedBackprop, 
    'Deconvolution': Deconvolution, 
    'GuidedGradCAM': GuidedGradCam, 
    'Lime': Lime, 
    'KernelShap': KernelShap, 
}

transformer = sys.argv[1]
xai_method = sys.argv[2]
protein_id = sys.argv[3]
sequence = sys.argv[4]
num_aminoacids = len(sequence)
EMBEDDING_DIM = 1024 if transformer != 'ankh' else 768

if transformer == 'bert':
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    ref_input_ids = [tokenizer.cls_token_id] + [tokenizer.pad_token_id] * num_aminoacids + [tokenizer.sep_token_id]
    wrapper = BertWrapper(model, xai_method) 
    gc_layer = wrapper.model.encoder.layer[-1].output.dropout
    embedding_layer = model.embeddings
    aminoacids_range = range(1, num_aminoacids + 1)
    shift = -1
elif transformer == 't5':
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    ref_input_ids = [tokenizer.pad_token_id] * num_aminoacids + [tokenizer.eos_token_id]
    wrapper = T5Wrapper(model.encoder, xai_method) 
    gc_layer = wrapper.model.dropout
    embedding_layer = model.encoder.embed_tokens
    aminoacids_range = range(num_aminoacids)
    shift = 0
elif transformer == 'ankh':
    model, tokenizer = ankh.load_base_model()
    sequence = list(sequence)
    ref_input_ids = [0] * num_aminoacids + [1] 
    wrapper = T5Wrapper(model.encoder, xai_method) 
    gc_layer = wrapper.model.dropout
    embedding_layer = model.encoder.embed_tokens
    aminoacids_range = range(num_aminoacids)
    shift = 0
    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

ids = tokenizer.batch_encode_plus([sequence], add_special_tokens=True, pad_to_max_length=False, is_split_into_words=True if transformer=='ankh' else False)
input_ids = torch.tensor([ids['input_ids'][0]]).to(device)
ref_input_ids = torch.tensor([ref_input_ids]).to(device)
inputs_embeds = embedding_layer(input_ids)
ref_embeds = embedding_layer(ref_input_ids)

attr = np.zeros((num_aminoacids, EMBEDDING_DIM, num_aminoacids, EMBEDDING_DIM))

if xai_method == 'GuidedGradCAM':
    method = GuidedGradCam(wrapper, gc_layer)
else:
    method = XAI_METHODS[xai_method](wrapper)
start = time.time()
for i in aminoacids_range:
    for j in range(EMBEDDING_DIM):
        if xai_method == 'IntegratedGradients':
            attributions = method.attribute(inputs=inputs_embeds, baselines=ref_embeds, additional_forward_args=(i, j), n_steps=20) 
        elif xai_method in ['DeepLift', 'GradientShap', 'Lime', 'KernelShap']: 
            attributions = method.attribute(inputs=inputs_embeds, baselines=ref_embeds, additional_forward_args=(i, j)) 
        else:
            attributions = method.attribute(inputs=inputs_embeds, additional_forward_args=(i, j))
        attr[i + shift][j] = attributions.squeeze(0).detach().cpu()[aminoacids_range, :]
        print(i, j)
        break
end = time.time()
print(end - start)
attr = np.float32(attr)
with lzma.open(f'{transformer}/embeddings interpretations/{protein_id}_{xai_method}.xz', 'wb') as f:
    pickle.dump(attr, f)
