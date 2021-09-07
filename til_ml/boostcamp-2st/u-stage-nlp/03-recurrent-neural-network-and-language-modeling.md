---
description: '210907'
---

# \(03강\) Recurrent Neural Network and Language Modeling

## 1. Basics of Recurrent Neural Networks \(RNNs\)

![](../../../.gitbook/assets/image%20%281083%29.png)

이전 time step 에서 계산한 $$ H_{t-1} $$을 입력으로 받아서 현재 time step의 $$ H_t $$를 출력으로 내어주는 구조이다. 이 때 매 time step에서 동일한 파라미터를 사용하기 때문에 동일한 모듈을 사용한다는 뜻에서 Recurrent 가 되었다.

RNN의 기호들이 의미하는 바는 다음과 같다.

![](../../../.gitbook/assets/image%20%281086%29.png)

특히 y는 h에서 생성되는 값으로 매 time step마다 생성될 수도 있고 마지막에만 생성될 수도 있다.

* machine translation은 매번 생성되고 문장의 긍정 표현은 마지막에만 생성된다.

보통,  h를 구할 때 hyper tangent를 사용한다.

![](../../../.gitbook/assets/image%20%281067%29.png)

여기서 실제로는 W가 x와 h\(t-1\)에 대해서 두 개 있다고 가정할 수도 있지만, 결국엔 내적해서 더하므로 실제로는 한 개의 W가 있다고 생각하고 x와 h\(t-1\)을 세로로 결합한 상태에서 곱할 수도 있다.

![](../../../.gitbook/assets/image%20%281078%29.png)



## 2. Types of RNNs

### One-to-One

![](../../../.gitbook/assets/image%20%281082%29.png)

* 사람의 키를 입력 받아 몸무게를 예측하는 모델
* time step이나 sequence가 없는 일반적인 형태를 도식화한 모습

### One-to-Many

![](../../../.gitbook/assets/image%20%281087%29.png)

* Image Captioning 같은 Task에 많이 사용된다.
* time step으로 이루어지지 않은 입력을 제공하며 이미지에 필요한 단어들을 순차적으로 생성한다.
* 위 그림에서는 입력이 첫번째 time step에서만 들어가므로 그림은 저렇게 그렸지만 실제로는 RNN 모델을 이용하므로 두번째 time step부터는 0으로 채워진 행렬 또는 텐서가 입력된다.

### Many-to-One

![](../../../.gitbook/assets/image%20%281077%29.png)

* Sentiment Classification Task에 이용
* 각 입력마다 Sequence Words를 받고, 각 단어를 통해 전체 문장의 감정을 분석하게 된다.

### Many-to-Many

![](../../../.gitbook/assets/image%20%281071%29.png)

* Machine Translation에 이용
* 마지막 sequence까지 입력을 받고 이 때 까지는 출력을 내놓지 않다가 마지막 step 에서 입력을 받은 뒤 출력을 준다.



또 다른 구조로서, 입력이 주어질 때 마다 출력이 되는 딜레이가 존재하지 않는 Task가 존재한다.

* Video classification on frame level 이나 POS tagging Task에 이용한다.
  * 비디오 분류는 각 비디오의 이미지 한장 한장이 어떤 의미를 갖는 지 분석한다. 예를 들면 각 신이 전쟁이 일어나는 신이다 라던가. 주인공이 등장하지 않는 신이다 라던가. 등등



## 3. Character-level Language Model

언어 모델은 기본적으로 주어진 문자열의 순서를 바탕으로 다음 단어가 무엇인지 알아내는 Task이다. 보다 심플한 예제로서 Character로 다룬다.

* 처음에는 중복을 제거해서 사전을 구축한다.
* 전체 길이만큼의 차원을 가지는 원핫벡터로 알파벳을 나타낸다.

![](../../../.gitbook/assets/image%20%281069%29.png)

이 때, bias도 사용하며 h0 는 영벡터로 초기화한다.

![](../../../.gitbook/assets/image%20%281085%29.png)

Output 벡터는 다음과 같이 구할 수 있다.

![](../../../.gitbook/assets/image%20%281063%29.png)

이후, 소프트 맥스를 거쳐 가장 큰 값으로 output을 결정하게 되며 특정 문자에 1을 몰아준 Ground Truth와의 오차를 통해 back propagation이 이루어지게 된다.

![](../../../.gitbook/assets/image%20%281064%29.png)

이 때 inference 하는 과정은 다음과 같다.

* 첫번째 글자를 주고 얻은 output을 두번째 입력으로 설정한다. 이를 반복해서 모든 단어를 얻는다.

![](../../../.gitbook/assets/image%20%281084%29.png)

위 글은 셰익스피어의 희곡 중 하나이다. 자세히 보면 단순히 알파벳 뿐만 아니라 공백과 문장부호까지도 이어지는 것을 알 수 있다. 실제로 이러한 것까지 고려해야한다.

![](../../../.gitbook/assets/image%20%281074%29.png)

처음에는 잘 학습하지 못해 말도 되지 않는 엉터리 단어들을 내놓다가 학습을 거듭할 수록 말이 되는 문장이 되는 것을 알 수 있다.

RNN으로 논문을 생성하거나, 연극 대본 또는 프로그래밍 코드까지도 생성할 수 있다.



### BackPropagation through time, BPTT

![](../../../.gitbook/assets/image%20%281075%29.png)

수 천, 수 만 이상의 데이터를 학습하다보면 입력으로 제공되는 Sequence가 매우 길 수 있고 이에 따라 모든 output을 종합해서 Loss 값을 얻고 BackPropagtaion을 진행해야 한다. 현실적으로 이 길이가 길어지면 한꺼번에 처리할 수 있는 정보나 데이터의 양이 한정된 리소스 안에 담기지 못하기 때문에 제한된 길이의 Sequence 만을 학습하는 방법을 채택한다.

![](../../../.gitbook/assets/image%20%281089%29.png)

다음 이미지는 Hidden state의 특정한 dimension을 관찰해서 크기가 커지면 푸른색으로 크기가 음수로 작아지면 붉은색으로 표현했다.

![](../../../.gitbook/assets/image%20%281068%29.png)

이렇게 한 개의 dimension을 여러개 관찰하다가 특정 위치에서 흥미로운 패턴을 보게된다 큰 따옴표부터 다음 큰 따옴표까지 푸른색을 유지하다가 따옴표가 닫힌 이후부터는 붉은색을 유지하고 다시 따옴표를 만나면 푸른색으로 바뀌는 모습을 볼 수 있다.

![](../../../.gitbook/assets/image%20%281088%29.png)

* 즉, 이 차원에서는 큰 따옴표의 시작과 끝을 기억하는 용도로 사용되었음을 알 수 있다.

또, 다음 이미지는 프로그램 코드를 나타낸다.

![](../../../.gitbook/assets/image%20%281076%29.png)

* 이 셀은 해당 구문이 if문이라는 것을 기억한다는 것을 알 수 있다.

사실 이러한 특징은 바닐라 RNN이 아닌 LSTM이나 GRU를 사용했을 때의 결과이다. 간단한 구조의 RNN은 정작 많이 사용하지 않는데, 다음과 같은 문제가 발생하기 때문이다.

![](../../../.gitbook/assets/image%20%281080%29.png)

매 타임스텝마다 히든스테이트에 동일한 W가 곱해지다보니 등비수열의 꼴로 나타나지게 되고 여기서 공비가 1보다 작으면 Vanishing Gradient 문제가, 1보다 크면 Exploding Gradient 문제가 발생하게 된다.

![](../../../.gitbook/assets/1%20%282%29.gif)

위의 숫자는 backpropagtaion 될 때 W의 gradient값을 나타내며 점점 작아지고 있다. 0에 가까워질수록 유의미한 signal을 뒤쪽으로 전달할 수 없게된다. 회색은 0을 의미하고 그 외의 값은 0보다 크거나 작은 값을 의미한다. RNN은 쉽게 gradient가 0이 되는 반면에 LSTM은 꽤 긴 타임 스텝까지도 gradient가 살아있는 모습이다.

* Long Term Dependency를 해결해주는 모습.



## 실습

### 필요 패키지

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
```

* 현재는 데이터의 길이가 모두 다른 모습

```python
max_len = len(max(data, key=len))
print(f"Maximum sequence length: {max_len}")

valid_lens = []
for i, seq in enumerate(tqdm(data)):
  valid_lens.append(len(seq))
  if len(seq) < max_len:
    data[i] = seq + [pad_id] * (max_len - len(seq))
```

* 가장 긴 데이터의 길이로 통일해준다.
* 이 때 이 길이보다 짧은 데이터는 `pad_id==0` 으로 채워준다.

```python
for i in data:
  print(i)
print(valid_lens)
```

```text
[85, 14, 80, 34, 99, 20, 31, 65, 53, 86, 3, 58, 30, 4, 11, 6, 50, 71, 74, 13]
[62, 76, 79, 66, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[93, 77, 16, 67, 46, 74, 24, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[19, 83, 88, 22, 57, 40, 75, 82, 4, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[70, 28, 30, 24, 76, 84, 92, 76, 77, 51, 7, 20, 82, 94, 57, 0, 0, 0, 0, 0]
[58, 13, 40, 61, 88, 18, 92, 89, 8, 14, 61, 67, 49, 59, 45, 12, 47, 5, 0, 0]
[22, 5, 21, 84, 39, 6, 9, 84, 36, 59, 32, 30, 69, 70, 82, 56, 1, 0, 0, 0]
[94, 21, 79, 24, 3, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[80, 80, 33, 63, 34, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[87, 32, 79, 65, 2, 96, 43, 80, 85, 20, 41, 52, 95, 50, 35, 96, 24, 80, 0, 0]

[20, 5, 8, 10, 15, 18, 17, 6, 6, 18]
```

* 길이가 모두 20으로 통일된 모습
* 또, 원래 길이를 알기위한 valid\_lens 를 선언한다.

```python
# B: batch size, L: maximum sequence length
batch = torch.LongTensor(data)  # (B, L)
batch_lens = torch.LongTensor(valid_lens)  # (B)
```

* 데이터를 하나의 batch로 만든다.
  * 단순히 Tensor화 해주는 과정으로 생각하면 된다.

### RNN 사용

RNN에 넣기전에 워드 임베딩을 해야한다.

```python
embedding_size = 256
embedding = nn.Embedding(vocab_size, embedding_size)

# d_w: embedding size
batch_emb = embedding(batch)  # (B, L, d_w)
```

* 처음에 vocab\_size는 100으로 정해주었다.
  * 실제로 각각의 데이터의 원소는 0부터 99까지의 수로 이루어져있다.
* embedding 차원은 256으로 임의로 정한다.

```python
hidden_size = 512  # RNN의 hidden size
num_layers = 1  # 쌓을 RNN layer의 개수
num_dirs = 1  # 1: 단방향 RNN, 2: 양방향 RNN

rnn = nn.RNN(
    input_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=True if num_dirs > 1 else False
)

h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h)
```

* hidden\_laye를 정의했다.
  * 단어는 100차원-&gt;256차원 -&gt; 512차원 -&gt; 256차원-&gt;100차원으로 변환된다.
  * 100차원은 원핫 인코딩
  * 256차원은 워드 임베딩
  * 512차원은 RNN network를 통해 변환
* 초기 h0은 0으로 초기화된다. 크기는 \(1, 10, 512\) 이다.

이후, RNN에 batch data를 넣는다. 두 가지 output을 얻는다.

```python
hidden_states, h_n = rnn(batch_emb.transpose(0, 1), h_0)

# d_h: hidden size, num_layers: layer 개수, num_dirs: 방향의 개수
print(hidden_states.shape)  # (L, B, d_h)
print(h_n.shape)  # (num_layers*num_dirs, B, d_h) = (1, B, d_h)
```

```text
torch.Size([20, 10, 512])
torch.Size([1, 10, 512])
```

* `transpose` 는 전치행렬이며 인자로 받은 차원끼리 변경시켜준다.
  * 현재는 0과 1이므로 X\_ij 에서 i가 j로, j가 i로 바뀐다.
* `hidden_states`: 각 time step에 해당하는 hidden state들의 묶음.
* `h_n`: 모든 sequence를 거치고 나온 마지막 hidden state.

### RNN 활용

마지막 hidden state를 이용하면 text classification task에 적용할 수 있다.

```python
num_classes = 2
classification_layer = nn.Linear(hidden_size, num_classes)

# C: number of classes
output = classification_layer(h_n.squeeze(0))  # (1, B, d_h) => (B, C)
print(output.shape)
```

```text
torch.Size([10, 2])
```

각 time step에 대한 hidden state를 이용하면 token-level의 task를 수행할 수도 있다.

```python
num_classes = 5
entity_layer = nn.Linear(hidden_size, num_classes)

# C: number of classes
output = entity_layer(hidden_states)  # (L, B, d_h) => (L, B, C)
print(output.shape)
```

```text
torch.Size([20, 10, 5])
```

### PackedSequence 사용

주어진 data에서 불필요한 pad 계산이 포함되기 때문에 정렬을 해야한다.

* 왜 불필요하냐면 0에다가 어떤 수를 곱해도 늘 0이므로 이 부분을 계산할 필요가 없는 것이다.
* 이게 문제가 돼? 문제가 된다. 각 타임스텝 마다 계산이 필요한데 이 때 많은 0을 계산하는 것보다 생략하는 것이 더 빠른 연산을 처리할 수 있다.
* 정렬을 하면 해결돼? 해결이 된다. 정렬을 하고 각 타임스텝별로 문장의 최대 길이를 기억하고 있으면 된다.
* 이 때 이 기능은 `torch.nn.utils.rnn` 에서 지원하는 `pack_padded_sequence` 와 `pad_packed_sequence` 를 이용하면 된다.
* [여기](https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html)를 보면 이해가 쉽다

![](../../../.gitbook/assets/image%20%281065%29.png)

![https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html](../../../.gitbook/assets/image%20%281072%29.png)

```python
sorted_lens, sorted_idx = batch_lens.sort(descending=True)
sorted_batch = batch[sorted_idx]

sorted_batch_emb = embedding(sorted_batch)
packed_batch = pack_padded_sequence(sorted_batch_emb.transpose(0, 1), sorted_lens)

packed_outputs, h_n = rnn(packed_batch, h_0)

outputs, outputs_lens = pad_packed_sequence(packed_outputs)

print(outputs.shape)  # (L, B, d_h)
print(outputs_lens)
```

```text
torch.Size([20, 10, 512])
tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])
```

* 주어진 data를 정렬하고 embedding한 뒤

   정렬한 data를 임베딩하고 PackSequence 모양으로 바꿔서 rnn에 입력한다.

* 이후 얻은 output은 원래 output형태와 다르므로 `pad_packed_sequence` 를 이용하여 원래 형태로 되돌려 준다.





