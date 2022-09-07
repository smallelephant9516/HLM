import os, sys
import numpy as np
import torch
import itertools
import random, math
import json
from collections import Counter
import time

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast,RobertaConfig,RobertaForMaskedLM, \
    LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, \
    pipeline


vocabulary = set(itertools.chain.from_iterable(corpus_ignore))
vocabulary_size = len(vocabulary)
print(vocabulary_size)

word_to_index = {w: idx for (idx, w) in enumerate(vocabulary)}
index_to_word = {idx: w for (idx, w) in enumerate(vocabulary)}


output_path='./'
paths = [str(x) for x in Path(output_path+"/corpus/").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=vocabulary_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
if os.path.isdir(output_path+"/tokens") is False:
    os.mkdir(output_path+"/tokens")
if os.path.isdir("./tokens") is False:
    os.mkdir("./tokens")
tokenizer.save_model(output_path+"/tokens")
tokenizer.save_model("./tokens")


with open(output_path+'/tokens/vocab.json') as f:
    decode = json.load(f)
encode={value:key for (key, value) in decode.items()}

corpus_code=[]
for i in range(len(corpus_ignore)):
    lst=corpus_ignore[i]
    corpus_row=[]
    for j in range(len(lst)):
        corpus_row.append(encode[word_to_index[lst[j]]+5])
    corpus_code.append(corpus_row)

with open(output_path+"/corpus/helical.txt","w") as f:
    for i in range(len(corpus_code)):
        lst=corpus_code[i]
        for j in range(len(lst)):
            if j==len(lst)-1:
                f.write(lst[j]+'\n')
            else:
                f.write(lst[j])

if '0' in [i for i in vocabulary]:
    del decode[encode[word_to_index['0']+5]]
    print('delete the ignored group',print(encode[word_to_index['0']+5]))
with open('./tokens/vocab.json','w') as f:
    json.dump(decode,f)
with open(output_path+'/tokens/vocab.json','w') as f:
    json.dump(decode,f)

tokenizer = RobertaTokenizerFast.from_pretrained(output_path+"/tokens", max_len=514)



config = RobertaConfig(
    vocab_size=50_000,
    max_position_embeddings=128,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
    position_embedding_type=None
)

model = RobertaForMaskedLM(config=config)

data_import = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=output_path+"/corpus/helical.txt",
    block_size=block_size,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    train_dataset=data_import
)

trainer.train()

trainer.save_model(output_path+"/tokens/")
trainer.save_model("./tokens/")

feature_extraction = pipeline(
    'feature-extraction',model="./tokens",tokenizer="./tokens")

def cut_corpus(corpus,cut_length):
    cut_index=[]
    new_corpus=[]
    cut_length=cut_length
    print(len(corpus))
    for i in range(len(corpus)):
        lst=corpus[i]
        n=len(lst)
        if n<=cut_length:
            new_corpus.append(lst)
            continue
        if n%cut_length==0:
            cut_amount=int(n/cut_length)
        else:
            cut_amount=int((n-n%cut_length)/cut_length)+1
        for j in range(cut_amount-1):
            cut_index.append(i)
            new_corpus.append(lst[j*cut_length:(j+1)*cut_length])
        new_corpus.append(lst[(cut_amount-1)*cut_length:])
    print(len(new_corpus))
    return new_corpus,cut_index
corpus_code_cut,cut_index=cut_corpus(corpus_code,block_size-2)

filament_embeddings=[]
for i in range(len(corpus_code_cut)):
    if i%200==0:
        print(i)
    lst=list(np.squeeze(feature_extraction(''.join(corpus_code_cut[i])))[0])
    filament_embeddings.append(lst)