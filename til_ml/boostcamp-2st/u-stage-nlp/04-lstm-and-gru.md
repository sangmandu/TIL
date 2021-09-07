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

![](../../../.gitbook/assets/image%20%281106%29.png)

* 여기서 x는 x\_t 이고 h는 h\_\(t-1\) 이다.

#### Forget gate

![](../../../.gitbook/assets/image%20%281103%29.png)

이전 타임스텝에서 얻은 정보 중 일부만을 반영하겠다.

= 이전 타임스텝에서 얻은 정보 일부를 까먹겠다 = forget

#### Input gate

![](../../../.gitbook/assets/image%20%281105%29.png)

이번 셀에서 얻은 C tilda 값을 input gate와 곱해주는 이유는 다음과 같다.

* 한번의 선형변환만으로 $$ C_{t-1} $$에 더해주는 정보를 만들기가 어렵다. 따라서 이 더해주는 정보를 일단 크게 만든 후에 각 차원별로 특정 비율만큼 덜어내서 더해주는 정보를 만들겠다 라는 목적이다.
* 이 때, 더해주는 정보보다 크게 만든 정보가 C tilda 이며 특정 비율만큼 덜어내는 작업이 input gate와 곱해주는 작업이다.

#### Output gate

![](../../../.gitbook/assets/image%20%281098%29.png)

* "He said, 'I love you.' " 라는 문장이 있다고 하자. 현재 sequence가 love y 까지 들어왔고 y다음에 o를 출력으로 줘야 할 차례이다. 이 때 y의 입장에서는 당장의 작은 따옴표가 열린 사실은 중요하지 않지만, 계속 전달해줘야하는 정보이다. 그래서 Ct의 activate function을 거친값을 o\_t에 곱해주는 것으로 해석할 수 있다.



### Gated Recurrent Unit \(GRU\)

![](../../../.gitbook/assets/image%20%281101%29.png)

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

```python
# d_w: word embedding size
batch_emb = embedding(batch)  # (B, L, d_w)

packed_batch = pack_padded_sequence(batch_emb.transpose(0, 1), batch_lens)

packed_outputs, (h_n, c_n) = lstm(packed_batch, (h_0, c_0))
print(packed_outputs)
print(packed_outputs[0].shape)
print(h_n.shape)
print(c_n.shape)
```

```text
PackedSequence(data=tensor([[-0.0690,  0.1176, -0.0184,  ..., -0.0339, -0.0347,  0.1103],
        [-0.1626,  0.0038,  0.0090,  ..., -0.1385, -0.0806,  0.0635],
        [-0.0977,  0.1470, -0.0678,  ...,  0.0203,  0.0201,  0.0175],
        ...,
        [-0.1911, -0.1925, -0.0827,  ...,  0.0491,  0.0302, -0.0149],
        [ 0.0803, -0.0229, -0.0772,  ..., -0.0706, -0.1711, -0.2128],
        [ 0.1861, -0.1572, -0.1024,  ..., -0.0090, -0.2621, -0.2803]],
       grad_fn=<CatBackward>), batch_sizes=tensor([10, 10, 10, 10, 10,  9,  7,  7,  6,  6,  5,  5,  5,  5,  5,  4,  4,  3,
         1,  1]), sorted_indices=None, unsorted_indices=None)
torch.Size([123, 512])
torch.Size([1, 10, 512])
torch.Size([1, 10, 512])
```

* hidden state와 cell state의 크기가 같은것을 볼 수 있다. 
* packed\_outputs 의 사이즈가 123인 이유를 아는가? 사실은 200이어야 한다. 여기서 0의 개수를 빼면 123이된다!

```python
outputs, output_lens = pad_packed_sequence(packed_outputs)
print(outputs.shape)
print(output_lens)
```

```text
torch.Size([20, 10, 512])
tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])
```



### GPU 사용

GPU는 Cell state가 없다. 그 외에는 동일하다.

```python
gru = nn.GRU(
    input_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=True if num_dirs > 1 else False
)

output_layer = nn.Linear(hidden_size, vocab_size)

input_id = batch.transpose(0, 1)[0, :]  # (B)
hidden = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (1, B, d_h)
```

Teacher forcing 없이 이전에 얻은 결과를 다음 input으로 이용한다.

* Teacher forcing이란, Seq2seq\(Encoder-Decoder\)를 기반으로 한 모델들에서 많이 사용되는 기법이다.  아래 설명과 이미지는 [여기](https://blog.naver.com/PostView.naver?blogId=sooftware&logNo=221790750668)를 참고했다.

![](../../../.gitbook/assets/image%20%281100%29.png)

* t-1번째의 디코더 셀이 예측한 값을 t번째 디코더의 입력으로 넣어준다. t-1번째에서 정확한 예측이 이루어진다면 엄청난 장점을 가지는 구조지만, 잘못된 예측 앞에서는 엄청난 단점이 되어버린다.
* 다음은 단점이 되어버린 RNN의 잘못된 예측이 선행된 경우

![](../../../.gitbook/assets/image%20%281099%29.png)

* 이러한 단점은 학습 초기에 학습 속도 저하의 요인이 되며 이를 해결하기 위해 나온 기법이 티쳐포싱이다.

![](../../../.gitbook/assets/image%20%281102%29.png)

* 위와 같이 입력을 Ground Truth로 넣어주게 되면, 학습시 더 정확한 예측이 가능하게 되어 초기 학습 속도를 빠르게 올릴 수 있다.
* 그러나 단점으로는 노출 편향 문제가 있다. 추론 과정에서는 Ground Truth를 제공할 수 없기 때문에 학습과 추론 단계에서의 차이가 존재하게 되고 이는 모델의 성능과 안정성을 떨어뜨릴 수 있다.
* 다만 노출 편향 문제가 생각만큼 큰 영향을 미치지 않는다는 연구결과가 있다.

> \(T. He, J. Zhang, Z. Zhou, and J. Glass. Quantifying Exposure Bias for Neural Language Generation \(2019\), arXiv.\)

```python
for t in range(max_len):
  input_emb = embedding(input_id).unsqueeze(0)  # (1, B, d_w)
  output, hidden = gru(input_emb, hidden)  # output: (1, B, d_h), hidden: (1, B, d_h)

  # V: vocab size
  output = output_layer(output)  # (1, B, V)
  probs, top_id = torch.max(output, dim=-1)  # probs: (1, B), top_id: (1, B)

  print("*" * 50)
  print(f"Time step: {t}")
  print(output.shape)
  print(probs.shape)
  print(top_id.shape)

  input_id = top_id.squeeze(0)  # (B)
```



### 양방향 및 여러 layer 사용

```python
num_layers = 2
num_dirs = 2
dropout=0.1

gru = nn.GRU(
    input_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    bidirectional=True if num_dirs > 1 else False
)
```

* 여기서는 2개의 레이어 및 양방향을 사용한다. 그래서 hidden state의 크기도 \(4, Batchsize, hidden dimension\) 이 된다.

```python
# d_w: word embedding size, num_layers: layer의 개수, num_dirs: 방향의 개수
batch_emb = embedding(batch)  # (B, L, d_w)
h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h) = (4, B, d_h)

packed_batch = pack_padded_sequence(batch_emb.transpose(0, 1), batch_lens)

packed_outputs, h_n = gru(packed_batch, h_0)
print(packed_outputs)
print(packed_outputs[0].shape)
print(h_n.shape)
```

```text
PackedSequence(data=tensor([[-0.0214, -0.0892,  0.0404,  ..., -0.2017,  0.0148,  0.1133],
        [-0.1170,  0.0341,  0.0420,  ..., -0.1387,  0.1696,  0.2475],
        [-0.1272, -0.1075,  0.0054,  ..., -0.0152, -0.0856, -0.0097],
        ...,
        [ 0.2953,  0.1022, -0.0146,  ...,  0.0467, -0.0049, -0.1354],
        [ 0.1570, -0.1757, -0.1698,  ...,  0.0369, -0.0073,  0.0044],
        [ 0.0541,  0.1023, -0.1941,  ...,  0.0117,  0.0276,  0.0636]],
       grad_fn=<CatBackward>), batch_sizes=tensor([10, 10, 10, 10, 10,  9,  7,  7,  6,  6,  5,  5,  5,  5,  5,  4,  4,  3,
         1,  1]), sorted_indices=None, unsorted_indices=None)
torch.Size([123, 1024])
torch.Size([4, 10, 512])
```

* 실제로 히든 스테이트의 크기가 4로 시작하는 것을 알 수 있다. 또한, packed\_outputs 역시 256개가 아니라 1024개의 차원으로 이루어진 것을 알 수 있다.

```python
outputs, output_lens = pad_packed_sequence(packed_outputs)

print(outputs.shape)  # (L, B, num_dirs*d_h)
print(output_lens)
```

```text
torch.Size([20, 10, 1024])
tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])
```



