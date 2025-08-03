# based on this model: https://www.youtube.com/watch?v=UU1WVnMk4E8
# the 2 validation files are the val_split.txt and the memory.txt (non-constant). memory.txt is a glorified log which enables the model to remember for eternity.
# train_split.txt is the training data file.
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#tinker with these:
block_size=64
batch_size=128
max_iters=100
eval_iters=100
learning_rate = 3e-4
n_embd=384
n_head=8
n_layer =8
dropout = 0.2
gradient_accumulation_steps = 4
print("starting up... at:")
print(device)
chars=""
with open('data_mem/vocab.txt','r',encoding='utf-8') as f:
    text=f.read()
    chars=sorted(list(set(text)))
vocab_size=len(chars)
string_to_int={ch:i for i,ch in enumerate(chars)}
int_to_string={i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])
def get_random_chunk(split):
    filename=""
    if split == 'train':
        filename = "data_mem/train_split.txt"
    elif split == 'val':
        if os.path.getsize("data_mem/memory.txt")>32767:
            filename = random.choices(["data_mem/val_split.txt","data_mem/memory.txt"], [0.9,0.1])[0]
        else:
            filename = "data_mem/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data
def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
class LanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02) #torch.nn was here if error.
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
    def forward(self,index,targets=None):
        B,T=index.shape
        tok_emb=self.token_embedding_table(index)
        pos_emb=self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb+pos_emb
        x = self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits, loss
    def generate(self,index,max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:] #tweak this
            logits,loss=self.forward(index_cond)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            index_next=torch.multinomial(probs,num_samples=1)
            index = torch.cat((index,index_next),dim=1)
        return index
model=LanguageModel(vocab_size)
print('loading model parameters...')
with open('model_01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully!')
m=model.to(device)
def append_to_memory(input_text):
    with open('data_mem/memory.txt', 'a', encoding='utf-8') as f:
        f.write(input_text + '\n')
#context=torch.zeros((1,1),dtype=torch.long,device=device)
#generated_chars=decode(m.generate(context,max_new_tokens=500)[0].tolist())
#print(generated_chars)
print("initialization done, select mode (train,*):")
prompt = input("Mode:\n")
if prompt=="train":
    max_iters = int(input("For how much iters (default is 100):\n"))
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    for iter in range(max_iters):
        if iter%eval_iters==0:
            losses=estimate_loss()
            print(f"Losses: {losses}") 
        xb,yb=get_batch('train')
        logits,loss=model.forward(xb,yb)
        loss=loss/gradient_accumulation_steps
        loss.backward()
        if iter % gradient_accumulation_steps == gradient_accumulation_steps - 1:
            optimizer.zero_grad(set_to_none=True)
            optimizer.step()
    print(loss.item())

    with open('model_01.pkl','wb') as f:
        pickle.dump(model,f)
    print("model saved")
else:
    while True:
        prompt = input("('exit' to exit) Message:\n")
        if prompt == "exit":
            break
        append_to_memory(prompt)
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
        generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
        print(f'Completion:\n{generated_chars}')