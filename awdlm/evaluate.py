import argparse
import torch
from torch.autograd import Variable
import time
import math
import os
import hashlib
import model
from utils import batchify, get_batch, repackage_hidden
import numpy as np
from collections import OrderedDict
from collections import Counter
from splitcross import SplitCrossEntropyLoss

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

model_corpus_dir = os.environ.get('AWD_LM_DIR', '/home/ubuntu/awd-lstm-lm/')

# Model parameters.
parser.add_argument('--data', type=str, default='', help='location of the data corpus')
parser.add_argument('--model', type=str, default='QRNN', help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='model_0.pt', help='model checkpoint to use')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--bptt', type=int, default=20,help='sequence length')
parser.add_argument('--emsize', type=int, default=400,help='size of word embeddings')

args = parser.parse_args()
args.cuda = True



def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    
    hidden = model.init_hidden(batch_size)
    
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
    """
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)
    """
    return 1





with open( os.path.join(model_corpus_dir,args.checkpoint), 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage)[0]

model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

test_batch_size = 1

fn = 'corpus.8d62dad9d767f8b663ab8699acd7ab95.data'
print('Loading cached dataset...')
corpus = torch.load(os.path.join(model_corpus_dir,fn))

test_data = None
unk_token = 'UNKNOWNWORD'

with open(args.data, 'r') as f:
    ntokens = 0
    for line in f:
        words = line.split() + ['<eos>']
        for word in words:
            ntokens+=1

    test_data = torch.LongTensor(ntokens)
    token = 0

    for line in f:
        words = line.split() + ['<eos>']
        for word in words:
            if word in corpus.dictionary.word2idx:
                test_data[token] = corpus.dictionary.word2idx[word]
            else:
                test_data[token] = corpus.dictionary.word2idx[unk_token]
            token += 1


ntokens   = len(corpus.dictionary)
criterion = None

if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
if args.cuda:
    criterion = criterion.cuda()



torch.save(corpus, fn)
print(test_data.size())
test_data = batchify(test_data , test_batch_size, args)
print(test_data.size())
"""
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
#"""