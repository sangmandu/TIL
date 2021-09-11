---
description: '210908'
---

# \[필수 과제2\] RNN-based Language Model

### 데이터 업로드

부스트코스에서 제공된 wikitext-2.zip 파일을 사용한다. 내부는 `train.txt, valid.txt, test.txt` 로 이루어져 있다.

```python
path_train = './train.txt'
with open(path_train, 'r', encoding="utf8") as f:
    corpus_train = f.readlines()    

# train dataset 크기 확인
print(len(corpus_train))

# 처음 10 문장을 print 해 봅시다.
for sent in corpus_train[:10]:
    print(sent)
```

```text
36718
 

 = Valkyria Chronicles III = 

 

 Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . <unk> the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " <unk> Raven " . 

 The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more <unk> for series newcomers . Character designer <unk> Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . 

 It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 . 

 

 = = Gameplay = = 

 

 As with previous <unk> Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces . Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through <unk> text . The player progresses through a series of linear missions , gradually unlocked as maps that can be freely <unk> through and replayed as they are unlocked . The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player . Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs . Alongside the main story missions are character @-@ specific sub missions relating to different squad members . After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game . There are also love simulation elements related to the game 's two main <unk> , although they take a very minor role . 
```



### 1. 데이터 클래스 준비

라이브러리 임포트

```python
import os
from io import open
import torch
```



Dictionary Class

```python
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
```

* word를 입력하면 index를, index를 입력하면 word를 받을 수 있도록 설정한다.



Corpus Class

```python
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
```

* 말뭉치 클래스는 각 데이터들을 토크나이징 하고 이를 보캡에 추가한다.
* 토크나이징 방식은 단순히 공백문자를 기준으로 나누며 각 줄에 마지막에는 eos 토큰을 추가한다. 그리고 각 단어들을 딕셔너리에 넣는다.
* 이후,  vocab을 참조해서 각 문장의 단어를 인덱스화 해서 저장한다. 이들은 학습에 사용할 것이므로 `torch.tensor` 로 캐스팅해준다.



```python
# corpus 확인
path = './'
corpus = Corpus(path)

print(corpus.train.size())
print(corpus.valid.size())
print(corpus.test.size())
```

```text
torch.Size([2088628])
torch.Size([217646])
torch.Size([245569])
```

* 각 범주별 데이터의 수를 알 수 있다. 훈련 데이터는 검증 데이터와 평가 데이터의 10배 정도의 크기이다.



### Question

현재 코드는 train, valid, test 데이터를 모두 dictionary에 갖게 된다. 이 때 발생하는 문제점은 무엇일까?

* 
이를 해결하려면 코드를 어떻게 바꿔야 할까?



### 2. 모델 아키텍처 준비

라이브러리 임포트

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```



RNN Model Class

`init`

```python
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
```

* token의 개수 `ntoken` 을 입력받아 Embedding의 입력차원으로 구성한다. `ninp` 는 임베딩 차원을 의미하며 `nhid` 는 히든 차원을 의미한다.
* `LSTM, GRU, RNN_TANH, RNN_RELU` 에 대한 4가지 모듈을 사용할 수 있다.

`init_weights`

```python
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
```

* 인코더의 파라미터를 -0.1 부터 0.1 사이의 균등분포를 만족하도록 초기화한다.
* 디코더도 마찬가지.

`forward`

```python
    def forward(self, input, hidden):
        emb = self.encoder(input)
        emb = self.drop(x)
        output, hideen = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
       
       return F.log_softmax(decoded, dim=1), hidden
```



`init_hidden`

```python
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
```

* RNN 타입이 LSTM일 경우 Cell state와 Hidden state 두 개의 가중치를, 그외의 경우에는 Hidden state 한 개의 가중치를 반환한다.



### 3. 모델 학습

모델에 필요한 argument를 설정한다.

```python
import argparse
import time
import math
import os
import torch
import torch.nn as nn

# argparse 대신 easydict 사용
import easydict
args = easydict.EasyDict({
    "data"    : './',                   # location of the data corpus
    "model"   : 'RNN_TANH',             # type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
    "emsize"  : 200,                    # size of word embeddings
    "nhid"    : 200,                    # number of hidden units per layer
    "nlayers" : 2,                      # number of layers
    "lr"      : 20,                     # initial learning rate
    "clip"    : 0.25,                   # gradient clipping
    "epochs"  : 6,                      # upper epoch limit
    "batch_size": 20,                   # batch size
    "bptt"    : 35,                     # sequence length
    "dropout" : 0.2,                    # dropout applied to layers (0 = no dropout)
    "seed"    : 1111,                   # random seed
    "cuda"    : True,                   # use CUDA
    "log_interval": 200,                # report interval
    "save"    : 'model.pt',             # path to save the final model
    "dry_run" : True,                   # verify the code and the model

})

# 디바이스 설정
device = torch.device("cuda" if args.cuda else "cpu")
```



이후, 말뭉치 데이터를 특정 배치만큼 나눠준다.

```python
###############################################################################
# Load data
###############################################################################

corpus = Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
```

*  `torch.narrow` 는 input, dim, start, length 를 인자로 받으며, input의 dim차원에 대해 start부터 length까지만 슬라이싱한다.
  * 여기서는 가장 바깥쪽 차원에 대해서 처음부터 nbatch \* bsz 만큼까지 슬라이싱하는데, 이는 bsz로 데이터를 나누면 마지막에 남는 데이터가 생겨서 이를 제거해주기 위함이다.
* `view` 를 통해 transpose 해준다. 근데 그 뒤에 또 `.t()` 로 transpose 해준다...? 뭘까? 언뜻 보면 transpose의 transpose를 해준다고 보이지만 첫 data가 1차원이어서 이를 2차원으로 만들어주고 transpose를 해준다.
  * 아니, 그러면 애초에 `data.view(-1, bsz)` 를 해주면 되는거 아니야? 라고 또 할 수 있지만 둘의 원소 배열이 좀 다르다. 일단 배치 사이즈만큼으로 행을 구성하고 transpose를 해줘야 하며, 처음부터 배치 사이즈만큼으로 열을 구성하게 되면 시퀀스의 배치가 섞이게 된다.
* `contiguous()`는 텐서의 모양을 변형할 때 메모리에 텐서가 뒤죽박죽 쌓이게 된다고 한다. 변형해서 생성된 텐서를 마치 처음 생성한 것처럼 메모리에 쌓이게 해준다고 한다.
  * 변형될 때 메모리에 흩어진다는 뜻은 아니며, 메모리의 원소들의 순서가 뒤죽박죽이라는 이야기. 아무래도 메모리의 지역성을 고려해주는 것 같다. 순서가 뒤죽박죽이면 그로인해 생기는 delay가 있을테니!



```python
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).to(device)

criterion = nn.NLLLoss()
```

```python
###############################################################################
# Training code1 - define functions
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()

        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)

        ############################ ANSWER HERE ################################
        # TODO: train 함수를 (이곳에) 완성해주세요.
        #
        # Hint1: output 을 받았으니, loss를 계산할 차례입니다.
        #        loss 를 계산한 후 해야 하는 일은 무엇일까요?
        #########################################################################
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break
```

* `repackage_hidden`
  * 히든 스테이트를 다시 재구성 하는 작업이다. 재구성을 왜 해야될까? 한번 forward를 하고 backward를 하면 이 때 grad값이 hidden state에 남아있게된다. 그러면 두번째 학습에서 grad를 전파할 때 이전 grad값이 영향을 주게 된다. 그래서 `detach` 라는 함수를 써서 이를 초기화하는 것이다.
* `get_batch`
  * 현재는 train\_data 안에 X와 y값이 모두 들어있으므로 이를 data와 target으로 분리해주는 역할을 한다. 각 시퀀스는 바로 뒷 단어를 예측하면 되므로 인덱스를 +1 해서 슬라이싱 해준다.
* `evaluate`
  * `train` 과정과 동일하다. 아래에서 부가 설명
* `train`
  * 기존과는 다르게 `optimizer.zero_grad` 를 해주지 않고 `model.zero_grad` 를 해준다. 왜일까? 왜냐하면 optimizer가 없기 때문이다.
  * 그럼 optimizer는 없는거야? 아니다! 단순히 optimizer라는 모듈을 쓰지 않을 뿐이고
  * line 57 그리고 73-74에서 그 역할을 하게된다.
  * grad를 0으로 초기화해주는 repackage 작업을 거친뒤 모델에서 output과 hidden을 얻고 loss를 계산하고, 파라미터를 갱신해주는 작업을 거친다.



