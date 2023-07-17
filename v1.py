import torch
import math
from mecab import MeCab
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# create instance of mecab tokenizer
mecab = MeCab()

# korean for 1st and 2nd pronouns
pron = ["나", "너", "내"]


with open("train.csv", "r", encoding="utf-8") as f:
    text = f.read()
texttr = text.splitlines()[1:]

with open("dev.csv", "r", encoding="utf-8") as f:
    text = f.read()
textdev = text.splitlines()[1:]

linetr, Xtr, Ytr = [], [], []
wc_pos, wc_neg = {}, {}
linedev, Xdev, Ydev = [], [], []

# manually extracting features 
# x1: count of positive lexicon
# x2: count of negative lexicon
# x3: 1 if "no" appears, 0 otherwise; I used mecab.pos method, and looked for "VCN" (Verb, Copula, Negative)
# x4: count of appearances of 1st / 2nd pronouns
# x5: 1 if "!" appears, 0 otherwise
# x6: word count of document
# "feature_extraction" will return a list; this will have to be cast into torch.tensor later. 
def feature_extraction(line):
    wc = 0.0
    x = [0.0,0.0,0.0,0.0,0.0,0.0]
    for w in mecab.morphs(line):
        if w in wc_pos and w in wc_neg:
            if wc_pos[w] > wc_neg[w]:
                x[0] += 1.0 # x1
            else:
                x[1] += 1.0 # x2
        if mecab.pos(w) == 'VCN':
            x[2] = 1 # x3 do we need this feature?
        if w in pron:
            x[3] += 1.0
        if w == "!":
            x[4] = 1.0
        wc += 1
    x[5] = math.log(wc)

    return x

# preprocess / tokenization using mecab-ko; morpheme based
for line in texttr:
    l = line[2:]
    y = line[:1]
    linetr.append(l)
    if y == "1":
        Ytr.append([1.0])
    else:
        Ytr.append([0.0])
    if y == '0':
        for w in mecab.morphs(l):
            if w in wc_neg:
                wc_neg[w] += 1
            else:
                wc_neg[w] = 1
    else:
        for w in mecab.morphs(l):
            if w in wc_pos:
                wc_pos[w] += 1
            else:
                wc_pos[w] = 1
    
# building Xtr and Ytr
for line in linetr:
    Xtr.append(feature_extraction(line))
Xtr = torch.tensor(Xtr)
Ytr = torch.tensor(Ytr)

# building Xdev and Ydev
for line in textdev:
    l = line[2:]
    y = line[:1]
    linedev.append(l)
    if y == "1":
        Ydev.append([1.0])
    else:
        Ydev.append([0.0])
    Xdev.append(feature_extraction(line))
Xdev = torch.tensor(Xdev)
Ydev = torch.tensor(Ydev)

# hyperparameters
num_layers = 10
learning_rate = 1e-2
max_iters = 100000
eval_iters = 5000

torch.manual_seed(42)

# one layer classifier
class SentimentClassifier(nn.Module):
    def __init__(self, num_inputs):
        super(SentimentClassifier, self).__init__()
        self.linear = nn.Linear(num_inputs, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out
    
num_inputs = 6
model = SentimentClassifier(num_inputs)

# SGD used as optimizer; this is a room for improvement, as I can use Adam instead later.
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

lossi = []

for epoch in range(max_iters):
    outputs = model(Xtr)
    loss = criterion(outputs, Ytr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % eval_iters == 0:
        lossi.append(loss.item())
        print(f"Epoch {epoch+1}/{max_iters}, Loss: {loss.item()}")

@torch.no_grad()
def split_loss(split):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
    }[split]
    logits = model(x)
    loss = criterion(logits,y)
    print(split, loss.item())
    
print(split_loss('train'))
print(split_loss('val'))