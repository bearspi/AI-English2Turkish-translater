import time, os, torch, math, torch.onnx
from torch import nn

import data, model
from tqdm.auto import tqdm

device = torch.device('mps' if torch.has_mps else 'cpu')

seed = 81

train_batch_size = 20

dim_m = 100

dropout = 0.5

n_neuron = 200
n_layers = 2

epochs = 5

bptt = 32

data_path = "./data/WMT16"

torch.manual_seed(seed)

corpus = data.Corpus(data_path)

def batchify(data, bsz: int) -> torch.tensor:
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, train_batch_size)
valid_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)

model = model.Transformer(num_tokens=ntokens, dim_model=dim_m, num_heads=2, num_encoder_layers=n_layers, n_hid=n_neuron, dropout_percent=dropout).to(device)

criterion = nn.CrossEntropyLoss()

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]  # Align the target with the input data
    return data, target



def train_loop(lr: int):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    total_loss = 0.
    start_time = time.time()
    n_tokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0), bptt)):
        data, target = get_batch(train_data, i)
        optimizer.zero_grad()   
        output = model(data)
        output = output.view(-1, n_tokens)
        target = target.view(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss
        
        if batch % 400 == 0 and batch > 0:
            cur_loss = total_loss / 400
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt,
                elapsed * 1000 / 400, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate_loop(data_source):
    model.eval()
    total_loss = 0.
    start_time = time.time()
    n_tokens = len(corpus.dictionary)
    with torch.inference_mode():
        for i in range(0, data_source.size(0), bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.permute(0, 2, 1)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

lr = 1
best_val_loss = None

try:
    for epoch in tqdm(range(1, epochs+1)):
        epoch_start_time = time.time()
        train_loop(1)
        val_loss = evaluate_loop(valid_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open("model.pt", 'wb') as f:
                torch.save(model, f)
                best_val_loss = val_loss
        else:
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')