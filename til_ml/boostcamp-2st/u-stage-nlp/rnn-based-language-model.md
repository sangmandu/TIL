---
description: '210908'
---

# \[필수 과제\] RNN-based Language Model

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
* `view` 를 통해 transpose 해준다. 근데 그 뒤에 또 `.t()` 로 transpose 해준다...? 뭘까?
* `contiguous()`는 텐서의 모양을 변형할 때 메모리에 텐서가 뒤죽박죽 쌓이게 된다고 한다. 변형해서 생성된 텐서를 마치 처음 생성한 것처럼 메모리에 쌓이게 해준다고 한다.
  * 변형될 때 메모리에 흩어진다는 뜻은 아니며, 메모리의 원소들의 순서가 뒤죽박죽이라는 이야기. 아무래도 메모리의 지역성을 고려해주는 것 같다. 순서가 뒤죽박죽이면 그로인해 생기는 delay가 있을테니!



