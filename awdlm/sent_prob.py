import argparse
import torch
from torch.autograd import Variable
import data
import time
import os
import hashlib
import model
from utils import batchify, get_batch, repackage_hidden
import numpy as np
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

model_corpus_dir = os.environ.get('AWD_LM_DIR', '/home/ubuntu/awd-lstm-lm/')


# Model parameters.
parser.add_argument('--data', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='QRNN',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='model_0.pt',
                    help='model checkpoint to use')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

args.cuda = True

def print_d(var_str):
    var_str_type  = 'type(' + var_str +')'
    var_str_dtype = var_str +'.dtype'
    var_str_size  = var_str +'.size()'
    print(var_str_type,' ==> '  ,eval(var_str_type) )
    print(var_str_dtype,' ==> ' ,eval(var_str_dtype) )
    print(var_str_size,' ==> '  ,eval(var_str_size) )


with open( os.path.join(model_corpus_dir,args.checkpoint), 'rb') as f:
    #model, _, _ = torch.load(f)
    model = torch.load(f, map_location=lambda storage, loc: storage)[0]

model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

print('args.cuda = ', args.cuda)
#fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
fn = 'corpus.8d62dad9d767f8b663ab8699acd7ab95.data'
print('Loading cached dataset...')
corpus = torch.load(os.path.join(model_corpus_dir,fn))

ntokens = len(corpus.dictionary)
print(ntokens)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), requires_grad=False)

print(input.size())
if args.cuda:
    input.data = input.data.cuda()

hidden_0 = model.init_hidden(1)

unk_token = 'UNKNOWNWORD'
model_hist = OrderedDict()    # maps sentence to [log_prob, hidden, prev_x]
# get sent1 if (sent1-word) in model_hist then 
# return model_hist[(sent1-word)][0] + get_sent_logprob(last two words in sent1, model_hist[(sent1-word)][1]) 

def remove_last_word(sent):
    return sent.rsplit(' ', 1)[0]

def get_last_two_words(sent):
    return ' '.join(sent.split()[-2:])

def get_word_freq(word):
    if word == unk_token:
        return 0
    elif word in corpus.dictionary.word2idx:
        idx = corpus.dictionary.word2idx[word]
        return corpus.dictionary.counter[idx]
    else:
        return 0


def get_sent_logprob_batch(sent_batch):
    original_sentences     = sent_batch.copy()
    sent_batch    = []
    new_sent_ids  = []
    all_log_props = [0]*len(original_sentences)

    for i, sent in enumerate(original_sentences):
        if not(sent in model_hist):
            sent_batch.append(sent)
            new_sent_ids.append(i)
        else:
            all_log_props[i] = model_hist[sent][0]

    batch_size       = len(sent_batch)


    max_history_size = 3*5000 #assuming max beam width = 5000

    while( len(model_hist) > max_history_size ):
        model_hist.popitem(last=False)

    sent_batch_original = sent_batch.copy()
    log_probs  = [0]*batch_size
    #print('batch_size = ', batch_size)

    if not(batch_size):
        return all_log_props, None, None

    input_batch = Variable(torch.rand(1,batch_size).mul(ntokens).long(), requires_grad=False)
    if args.cuda:
        input_batch.data = input_batch.data.cuda()
    hidden_0 = model.init_hidden( batch_size )
    if args.model == 'QRNN': model.reset()
    hidden = hidden_0.copy()

    partially_new_hidden   = [None]*batch_size
    partially_new_prev_x   = [None]*batch_size
    
    """
    sent_batch_split = []
    for sent in sent_batch:
        sent_batch_split.append( sent.split() )
    original_sentences = sent_batch_split    
    """
    #print('hidden_0[0].size()',hidden_0[0].size())
    #print(len(hidden_0))
    #print(type(hidden_0))
    # ---------------------------------------------------
    # restructuring sentences
    model.rnns[0].prevX = torch.zeros([1,batch_size,400], dtype=torch.float32 )
    
    if args.cuda:
        model.rnns[0].prevX.data = model.rnns[0].prevX.data.cuda()

    for i in range(batch_size):
        
        if remove_last_word(sent_batch[i]) in model_hist and len(sent_batch[i].split()) > 2:
            #print('#'*10)
            log_probs[i]     = model_hist[ remove_last_word(sent_batch[i]) ][0]
            hidden_          = model_hist[ remove_last_word(sent_batch[i]) ][1]
            prev_x           = model_hist[ remove_last_word(sent_batch[i]) ][2]
            #print(type(model.rnns[0].prevX))
            #print(model.rnns[0].prevX.size())

            model.rnns[0].prevX[:,i,:] = prev_x[0]
            model.rnns[1].prevX = None
            model.rnns[2].prevX = None
            model.rnns[3].prevX = None
            for h, h_ in zip(hidden, hidden_):
                h[:,i,:] = h_

            sent_batch[i] = get_last_two_words(sent_batch[i])
    # ---------------------------------------------------
    sent_batch_split = []
    for sent in sent_batch:
        sent_batch_split.append( sent.split() )
    sent_batch = sent_batch_split

    max_length = max([len(s) for s in sent_batch])

    finished_sentences = []

    for i in range(max_length-1):
        #if model.rnn_type == 'QRNN': model.reset()
        words = [] 
        next_words = []
        for j, sent in enumerate(sent_batch):
            if i < (len(sent)-1):
                words.append( sent[i] )
                next_words.append( sent[i+1] )
                freq = min( get_word_freq(sent[i]), get_word_freq(sent[i+1]) )
                if freq < 200:
                    #print('freq < 10')
                    words.append( unk_token )
                    next_words.append( unk_token )
                    finished_sentences.append(j)
                    log_probs[j] = -1000                    
            else:
                words.append( unk_token )
                next_words.append( unk_token )
                finished_sentences.append(j)

        word_idxs      = [corpus.dictionary.word2idx[word] for word in words]
        next_word_idxs = [corpus.dictionary.word2idx[next_word] for next_word in next_words] 
        #print(word_idxs)
        for j in range(batch_size):
            input_batch.data[0,j].fill_(word_idxs[j])
        #print_d('input_batch')
        #print(input_batch)
        output, hidden = model(input_batch, hidden)
        output_flat = model.decoder(output).cpu()
        props = torch.nn.functional.softmax(output_flat,dim=1)
        #print(props.size())

        for j in range(batch_size):
            if not(j in finished_sentences):
                """
                prior = 0
                if get_word_freq(next_words[j]):
                    prior = np.log10( get_word_freq(next_words[j]) / 3287751 )
                else:
                    prior = 0
                """
                log_prob = torch.log10( props[j][next_word_idxs[j]] ) #+ prior
                log_probs[j] += log_prob.tolist()

        hidden = repackage_hidden(hidden)

    # ---------------------------------------------------
    # saving history
    for i, original_sentence in enumerate(sent_batch_original):
        sent_prev_x  = []

        for rnn in model.rnns:
            if rnn.prevX is None:
                sent_prev_x.append(None)
            else:    
                #print(rnn.prevX.size())
                #print(rnn.prevX.dtype)
                sent_prev_x.append(rnn.prevX[:,i,:])
        
        sent_hidden  = [h[:,i,:] for h in hidden]
        sent_logprob = log_probs[i]
        model_hist[original_sentence] = [sent_logprob, sent_hidden, sent_prev_x]
    # ---------------------------------------------------
    """
    [print(type(rnn.prevX)) for rnn in model.rnns]
    try:
        print('prev_x[0].size()   ',prev_x[0].size())
    except:
        print('type(prev_x[0])   ',type(prev_x[0]))
    """
    #print(log_probs)

    for i,lp in zip(new_sent_ids,log_probs):
        all_log_props[i] = lp

    #for s,l in zip(original_sentences, all_log_props):
    #    print(s,' ==> ', l)

    return all_log_props, None, None



def get_sent_logprob(sentence, hidden_prev = None, prev_x = None):
    start_time = time.time()

    # prev_x 400*4
    #for h in model_hist:
     #   print('h: ', h)
    
    if hidden_prev is None:
        if sentence in model_hist:
            return model_hist[sentence]
        
        if remove_last_word(sentence) in model_hist and len(sentence.split()) > 2:
            log_prob_sum, hidden, prev_x =  get_sent_logprob(get_last_two_words(sentence), 
                                                         model_hist[remove_last_word(sentence)][1],
                                                         model_hist[remove_last_word(sentence)][2])
            model_hist[sentence] = [log_prob_sum + model_hist[remove_last_word(sentence)][0], hidden, prev_x]
            return model_hist[sentence]
    
    hidden_0 = model.init_hidden(1)

    if hidden_prev is None:
        if args.model == 'QRNN': model.reset()
        hidden = hidden_0.copy()
    else:
        for rnn, p in zip(model.rnns, prev_x):
            rnn.prevX = p
        hidden = hidden_prev
        

    log_probs  = []
    sent = sentence.split()
    

    for i in range(len(sent)-1):
        #if model.rnn_type == 'QRNN': model.reset()
        word      = sent[i]
        next_word = sent[i+1]
        word_idx      = corpus.dictionary.word2idx[word]
        next_word_idx = corpus.dictionary.word2idx[next_word]

        input.data.fill_(word_idx)
        #print_d('input')
        output, hidden = model(input, hidden)
        output_flat = model.decoder(output).squeeze().cpu()
        props = torch.nn.functional.softmax(output_flat)

        log_prob = torch.log10( props[next_word_idx] )
        log_probs.append( log_prob.tolist() )
        hidden = repackage_hidden(hidden)
    #print(log_probs)
    #print([int(x) for x in log_probs])
    #print("time = " , int((time.time() - start_time)*1000), ' ms')
    
    
    prev_x = [rnn.prevX for rnn in model.rnns]

    if hidden_prev is None:
        model_hist[sentence] = [sum(log_probs), hidden, prev_x]

    return sum(log_probs), hidden, prev_x


"""
log_probs,_,_ = get_sent_logprob_batch([])

sentences_1 = [ '<eos> he always goes to school by bus',
                '<eos> he always goes to school by bus' ]

sentences_2 = ['<eos> he always goes to school by bus <eos>',
               '<eos> he always goes to school by bus <eos>',
               '<eos> he always goes to school by bus <eos>',
               '<eos> he always goes to school by bus <eos>',
               '<eos> he always goes to school by bus <eos>',
               '<eos> he always goes to school by bus <eos>',
               '<eos> he always goes to school by bus <eos>',
               '<eos> he always goes to school by bus <eos>',
               '<eos> he always goes to school by man <eos>']


#start_time = time.time()

#for sent in sentences:
#    print( sent,'-->', get_sent_logprob(sent)[0] )
#print("time = " , int((time.time() - start_time)*1000), ' ms')
start_time = time.time()

log_probs,_,_ = get_sent_logprob_batch(sentences_1)
log_probs,_,_ = get_sent_logprob_batch(sentences_2)

print(log_probs)
print('len(log_probs)', len(log_probs))
for i, sent in enumerate(sentences_2):
    print( sent,'-->', log_probs[i] )

print("time = " , int((time.time() - start_time)*1000), ' ms')
#"""

"""
'<eos> is are bad good that <eos>',
             '<eos> he always go to school by bus <eos>',
             '<eos> he always goes to downstairs by bus <eos>',
             '<eos> bus by school to goes always he <eos>',

             '<eos> he always goes to school by bus <eos> ',
             '<eos> he always goes to school by man <eos>',
             '<eos> is are bad good that <eos>',
             '<eos> he always go to school by bus <eos>',
             '<eos> he always goes to downstairs by bus <eos>',
             '<eos> bus by school to goes always he <eos>'
"""
