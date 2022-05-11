

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Imports     --
#···············································································
import argparse
from collections import Counter
import csv
import pandas as pd
import torch
import tqdm
import transformers

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


def parse_args():

        parser = argparse.ArgumentParser(description="Trimming Model / Tokenizer")

        parser.add_argument(
                '--base_model',
                type=str
                )

        args = parser.parse_args()
        return args


tokenizer = transformers.MT5Tokenizer.from_pretrained('google/mt5-large')
model = transformers.MT5ForConditionalGeneration.from_pretrained('google/mt5-large')

df_fr = pd.read_csv('fra-fr_web_2011_1M-sentences.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
df_fr.columns = ['idx', 'text']
cnt_fr= Counter()
for text in tqdm.tqdm(df_fr.text):
    cnt_fr.update(tokenizer.encode(text))
print(len(cnt_fr), len(cnt_fr)/tokenizer.vocab_size)


for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_fr.most_common(top)) / sum(cnt_fr.values()))


df_en = pd.read_csv('eng-uk_web-public_2018_1M-sentences.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
df_en.columns = ['idx', 'text']
cnt_en = Counter()
for text in tqdm.tqdm(df_en.text):
    cnt_en.update(tokenizer.encode(text))
print(len(cnt_en), len(cnt_en)/tokenizer.vocab_size)


for top in 10_000, 20_000, 30_000:
    print(top, sum(v for k, v in cnt_en.most_common(top)) / sum(cnt_en.values()))


new_tokens = set(range(1000))
for i, (k, v) in enumerate(cnt_en.most_common(10_000)):
    if k not in new_tokens:
        new_tokens.add(k)
for i, (k, v) in enumerate(cnt_ru.most_common(25_000)):
    if len(new_tokens) == 29_900:
        print(i, 'Russan tokens are included')
        break
    if k not in new_tokens:
        new_tokens.add(k)
for t in range(tokenizer.vocab_size - 100, tokenizer.vocab_size):
    new_tokens.add(t)

print(len(new_tokens))
kept_ids = sorted(new_tokens)

new_size = len(kept_ids)
new_emb = torch.nn.Embedding(new_size, model.shared.embedding_dim)
new_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=new_size, bias=False)
for new_id, old_id in enumerate(kept_ids):
    new_emb.weight.data[new_id] = model.shared.weight.data[old_id]
    new_head.weight.data[new_id] = model.lm_head.weight.data[old_id]
model.shared.weight = new_emb.weight
model.lm_head.weight = new_head.weight
model.config.__dict__['vocab_size'] = new_size
model.config.__dict__['_name_or_path'] = 'trimmed/ru-en-mt5-xl'



import sentencepiece_model_pb2 as spmp
smp = tokenizer.sp_model.serialized_model_proto()
m = spmp.ModelProto()
m.ParseFromString(smp)
print('the loaded model has pieces:', len(m.pieces))
new_pieces = [m.pieces[idx] for idx in kept_ids]
print('the new pieces:', len(new_pieces))


# replace the content of the first 30K pieces
for i, p in enumerate(new_pieces):
    m.pieces[i].piece = p.piece
    m.pieces[i].score = p.score
    m.pieces[i].type = p.type

# drop the remaining pieces
n = len(new_pieces)
for i in tqdm.trange(len(m.pieces) - n):
    m.pieces.pop(len(m.pieces) - 1)

print(len(m.pieces))
with open('new_sp.model', 'wb') as f:
    f.write(m.SerializeToString())

new_tokenizer = transformers.MT5Tokenizer('new_sp.model', extra_ids=0)

new_tokenizer.save_pretrained('ru-en-mt5-xl')
model.save_pretrained('ru-en-mt5-xl')
