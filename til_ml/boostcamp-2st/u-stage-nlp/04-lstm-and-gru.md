---
description: '210907'
---

# \(04강\) LSTM and GRU

## 4. Long Short-Term Memory \(LSTM\) \| Gated Recurrent Unit \(GRU\)

### Long Short-Term Memory \(LSTM\)

Vanila Model의 Gradient Exploding/Vanishing 문제를 해결하고 Long Term Dependency 문제를 개선한 문제이다.

기존의 RNN 식은 다음과 같다.

$$ h_t = f_w(x_t,\ h_{t-1}) $$

LSTM에는 Cell state라는 값이 추가되며 식은 다음과 같다.

$$ \{C_t,\ h_t\} = LSTM(x_t,\ C_{t-1},\ h_{t-1}) $$

Cell state가 hidden state보다 좀 더 완성된, 필요로 하는 정보를 가지고 있는 벡터이며 이 cell state 벡터를 한번 더 가공해서 해당 time step에서 노출할 필요가 있는 정보를 필터링 할 수 있는 벡터로도 생각할 수 있다.

cell state를 한번 더 가공한 hidden state 벡터는 현재 timestep 에서 예측값을 계산하는 output layer의 입력벡터로 사용한다.

![](../../../.gitbook/assets/image%20%281103%29.png)

* 여기서 x는 x\_t 이고 h는 h\_\(t-1\) 이다.

#### Forget gate

![](../../../.gitbook/assets/image%20%281100%29.png)

이전 타임스텝에서 얻은 정보 중 일부만을 반영하겠다.

= 이전 타임스텝에서 얻은 정보 일부를 까먹겠다 = forget

#### Input gate

![](../../../.gitbook/assets/image%20%281102%29.png)

이번 셀에서 얻은 C tilda 값을 input gate와 곱해주는 이유는 다음과 같다.

* 한번의 선형변환만으로 $$ C_{t-1} $$에 더해주는 정보를 만들기가 어렵다. 따라서 이 더해주는 정보를 일단 크게 만든 후에 각 차원별로 특정 비율만큼 덜어내서 더해주는 정보를 만들겠다 라는 목적이다.
* 이 때, 더해주는 정보보다 크게 만든 정보가 C tilda 이며 특정 비율만큼 덜어내는 작업이 input gate와 곱해주는 작업이다.

#### Output gate

![](../../../.gitbook/assets/image%20%281098%29.png)

* "He said, 'I love you.' " 라는 문장이 있다고 하자. 현재 sequence가 love y 까지 들어왔고 y다음에 o를 출력으로 줘야 할 차례이다. 이 때 y의 입장에서는 당장의 작은 따옴표가 열린 사실은 중요하지 않지만, 계속 전달해줘야하는 정보이다. 그래서 Ct의 activate function을 거친값을 o\_t에 곱해주는 것으로 해석할 수 있다.



### Gated Recurrent Unit \(GRU\)

![](../../../.gitbook/assets/image%20%281099%29.png)

LSTM의 모델 구조를 경량화해서 적은 메모리 요구량과 빠른 계산이 가능하도록 만든 모델이다. 가장 큰 특징은 LSTM은 Cell과 Hidden이 있는 반면에 GRU에서는 Hidden만 존재한다는 것이다. 그러나 GRU의 동작원리는 LSTM과 굉장히 동일하다.

* LSTM의 Cell의 역할을 GRU에서는 Hidden이 해주고 있다고 보면된다.
* GRU 에서는 Input Gate만을 사용하며 Forget Gate 자리에는 1 - Input Gate 값을 사용한다.



## 실습

### 필요 패키지 import

```python
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
```

### 데이터 전처리

```python
vocab_size = 100
pad_id = 0

data = [
  [85,14,80,34,99,20,31,65,53,86,3,58,30,4,11,6,50,71,74,13],
  [62,76,79,66,32],
  [93,77,16,67,46,74,24,70],
  [19,83,88,22,57,40,75,82,4,46],
  [70,28,30,24,76,84,92,76,77,51,7,20,82,94,57],
  [58,13,40,61,88,18,92,89,8,14,61,67,49,59,45,12,47,5],
  [22,5,21,84,39,6,9,84,36,59,32,30,69,70,82,56,1],
  [94,21,79,24,3,86],
  [80,80,33,63,34,63],
  [87,32,79,65,2,96,43,80,85,20,41,52,95,50,35,96,24,80]
]

max_len = len(max(data, key=len))

valid_lens = []
for i, seq in enumerate(tqdm(data)):
  valid_lens.append(len(seq))
  if len(seq) < max_len:
    data[i] = seq + [pad_id] * (max_len - len(seq))
    
# B: batch size, L: maximum sequence length
batch = torch.LongTensor(data)  # (B, L)
batch_lens = torch.LongTensor(valid_lens)  # (B)

batch_lens, sorted_idx = batch_lens.sort(descending=True)
batch = batch[sorted_idx]
```

* [이전 실습](03-recurrent-neural-network-and-language-modeling.md#undefined-2)과 동일하다.



### LSTM 사용

LSTM은 Cell state가 추가된다. shape는 hidden state와 동일하다.

```python
embedding_size = 256
hidden_size = 512
num_layers = 1
num_dirs = 1

embedding = nn.Embedding(vocab_size, embedding_size)
lstm = nn.LSTM(
    input_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=True if num_dirs > 1 else False
)

h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h)
c_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h)
```

* hidden state와 cell state는 0으로 초기화한다.











