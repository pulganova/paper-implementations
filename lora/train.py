import torch
import torch.nn as nn
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from lora import LoRALayer, make_hook

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

for param in model.parameters():
    param.requires_grad = False

tokenizer.pad_token = tokenizer.eos_token

d, k, r, alpha = 768, 768, 4, 4
batch_size = 8
LoRAs = []

for i in range(len(model.transformer.h)):
    LoRAs.append([LoRALayer(d,k,r,alpha),LoRALayer(d,k,r,alpha)])

for i in range(len(model.transformer.h)):
    model.transformer.h[i].attn.c_attn.register_forward_hook(make_hook(i,LoRAs))

params = itertools.chain(*[lora.parameters() for pair in LoRAs for lora in pair])
optimizer = torch.optim.Adam(params, lr=1e-4)

dataset = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
dataset = list(dataset)
dataset = [text for text in dataset if text['output'] != '']
dataset = [dataset[i * batch_size : (i + 1) * batch_size] for i in range(len(dataset)//batch_size)]

losses =[]
for batch in tqdm(dataset[:50]):
    optimizer.zero_grad()
    input_ids = tokenizer([text['output'] for text in batch],return_tensors='pt',padding=True, truncation=True)['input_ids']
    output = model(input_ids=input_ids, labels=input_ids)
    loss = output.loss
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

plt.plot(losses)
plt.savefig("loss.png")
torch.save([lora.state_dict() for pair in LoRAs for lora in pair], 'lora_weights.pt')
