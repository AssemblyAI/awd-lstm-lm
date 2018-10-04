# RNN based Language model

## Setup

``` 	
sudo python3 setup.py install
```     

change model_corpus_dir in /awd-lstm-lm/awdlm/sent_prob.py to point to 
the directory where the trained model (model_0.pt) and corpus 
(corpus.8d62dad9d767f8b663ab8699acd7ab95.data) are downloaded


##Example

``` 	
import sys; 
sys.path.insert(0, "/home/ubuntu/awd-lstm-lm/awdlm/")
from awdlm.sent_prob import get_sent_logprob, corpus, get_sent_logprob_batch

sent1 = 'he is good'
sent2 = 'he is he'
batch = [sent1, sent2]
log_prop, _, _ = get_sent_logprob_batch(batch)
print(log_prop)
``` 	







