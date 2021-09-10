---
description: '210908'
---

# \(05강\) Sequence to Sequence with Attention

## 1. Seq2Seq with attention Encoder-decoder architecture Attention mechanism

### Seq2Seq Model

앞서 배운 RNN의 구조 중 Many to Many에 해당하는 모델이다. 보통 입력은 word 단위의 문장이고 출력도 동일하다.

![](../../../.gitbook/assets/image%20%281111%29.png)

이 때, 입력 문장을 받는 모델을 인코더라고 하고 하나하나 답을 내놓는 부분을 디코더라고 한다. 인코더와 디코더는 서로 다른 RNN 모델이다. 그래서 파라미터를 공유하거나 하지 않는다. \(인코더와 디코더 각각은 내부적으로 공유한다\)

또한, 내부 구조를 자세히 보면 LSTM을 채용한 것을 알 수 있다. 인코더의 마지막 단어까지 읽은 후 생성되는 마지막 스텝의 Hidden state는 디코더의 h0로서의 역할을 한다. 이 hidden state는 입력에 대한 정보를 잘 가지고 있다고 볼 수 있고 이를 바탕으로 디코더에서 사용한다고 볼 수 있다.

&lt;Start&gt; 토큰 또는 &lt;SoS&gt; \(Start of Sentence\) 토큰이 입력되면서 디코더가 작동되기 시작하며 &lt;End&gt; 토큰 또는 &lt;EoS&gt; \(End of Sentence\) 토큰이 나올 때 까지 디코더 RNN을 구동한다. 

Hidden state의 크기는 처음에 고정하기 때문에 아무리 짧은 문장이라도 hidden dimension만큼의 정보를 저장해야 하고, 아무리 긴 문장이라도 hidden dimnesion 만큼으로 정보를 압축해야 한다.

또, LSTM이 Long Term Dependency를 해결했다고 하더라도 훨씬 이전에 나타난 정보는 변질되거나 소실된다. 그래서 문장이 길다보면 첫번째 단어에 대한 정보가 적기 때문에 디코더의 시작부터 품질이 나빠지는 문제가 발생한다. 이에 대한 테크닉으로 "I go home" 으로 입력하는 것이 아닌 "home go I"로 입력해서 문장의 초반 정보를 잘 유지할 수 있도록 한다.

디코더는 인코더에서 마지막으로 나온 hIdden state를 h0으로 사용하지만 이것만을 사용하지 않는다. 인코더의 각 time step에서 나온 hidden state를 모두 제공받고 이 중 선별적으로 사용해서 예측에 도움을 주는 형태로 활용한다. 이것이 attention 모듈의 기본적인 아이디어이다.



### Seq2Seq Model with Attention

![](../../../.gitbook/assets/image%20%281108%29.png)

hidden state가 4개의 차원으로 구성되었고 프랑스어를 영어로 변환하는 과정을 예시로 든 이미지이다. 다음과 같은 순서로 구성된다.

* 인코더에서 입력별로 hidden state가 생성되며 최종 hidden state가 디코더에 제공된다.
* 디코더는 h0와 &lt;sos&gt; 토큰을 가지고 첫번째 h state를 생성한다.
* 첫번째 h state는 인코더의 각각의 h state와 내적을 하게 된다.
  * 내적을 한다는 것은 유사도를 비교하겠다는 의미.
* 이후, 각 유사도를 sofrmax한 값을 가중치로 얻게된다.
* 이 때 attention output 벡터는 가중평균된 벡터이며 context 벡터라고도 부른다.

![](../../../.gitbook/assets/image%20%281118%29.png)

* 이후 디코더는 디코더의 h state와 attention output 을 concat 하며 예측값을 반환하게된다.

![](../../../.gitbook/assets/image%20%281107%29.png)

* 마찬가지로, 디코더의 두번째 step에서도 동일한 메커니즘이 적용된다.
* &lt;eos&gt; 토큰이 나올때까지 작동된다.

정리하면 RNN의 디코더는 1\) 다음 단어를 예측하고 2\) 인코더로부터 필요로 하는 정보를 취사선택하도록, 학습이 진행된다. 역전파에 관점에서도, Attention 벡터가 다시 선택될 수 있도록 인코더의 hidden state가 갱신된다. 인코더의 h state가 갱신되므로 당연히 디코더의 h state도 갱신된다.

학습을 할 때에는 디코더의 각 타임스텝의 예측값이 무엇이든 간에 Ground Truth 값을 넣어주게 되지만 추론을 할 때에는 이전 타임스텝의 예측값을 다음 타임스텝의 입력값으로 넣어주게 된다.

* 이렇게 학습 중에 입력을 Ground Truth로 넣어주는 방법을 `Teacher Forcing` 이라고 한다.
* 물론, 학습은 잘 되지만 실제로 우리가 적용해야 하는 문제는 `Teacher Forcing` 과는 괴리가 있다. 그래서 이를 섞어서 사용하는 방법이 나왔는데, 학습 초반에는 빠른 학습을 위해서 이를 적용했다가, 학습이 어느 정도 되고나서는 적용하지 않도록 하는 방법도 존재한다.

### Different Attention Mechanisms

이전에는 유사도를 구하기 위해 내적을 사용했는데, 내적 이외에도 다양한 방법으로 attention을 구성하는 방법을 알아보도록 한다.

![](../../../.gitbook/assets/image%20%281116%29.png)

* h\_t : 디코더에서 주어지는 히든 벡터
* h\_s : 인코더에서 각 워드별로의 히든 벡터

그냥 내적을 할 수도 있지만 `generalized dot product` 라는 attention 방법도 있다.

* W는 대각행렬의 모양이다. 각 dimension 별로 적용하는 가중치의 역할을 한다.

또, `concat` 하는 방법이 있는데, 이전의 내적들과는 다른 방법이다. 유사도를 내적이 아니라 신경망을 통해서 구하는 방법이다.

![&#xB0B4;&#xAC00; &#xADF8;&#xB9B0; &#xAE30;&#xB9B0; &#xADF8;&#xB9BC;](../../../.gitbook/assets/image%20%281115%29.png)

![](../../../.gitbook/assets/image%20%281119%29.png)

여기서 W2에 해당하는 부분이 $$ v_a^T $$가 된다.

* 2-layer의 신경망으로 구성할 수 있다.

이전의 attention은 파라미터가 필요없는 내적 연산의 모듈이었는데, 파라미터가 필요한 학습이 되면서 좀 더 최적화 할 수 있게된다.



### Attention is Great

* 디코더의 매 스텝마다 특정 정보를 제공하면서 성능이 매우 향상되었다.
* attention을 하면서 긴 문장의 번역이 어려운 점, bottleneck problem을 해결했다.
* 역전파 과정에서 디코더 스텝과 인코더 스텝을 거쳐가면서 매우 긴 타임스텝을 지나게되고 이 때 gradient 소실 또는 증폭 문제가 발생할 수 있게되는데 attention을 사용하면서 gradient가 직접적으로 전달할 수 있는 방법이 추가되면서 gradient가 변질없이 전달될 수 있게되었다.
* 흥미로운 해석가능성을 제공해준다.
  * attention을 조사해서 h state가 각 단어의 어떤 부분에 집중했는지 관찰할 수 있게되었다.



## 실습

매 실습마다 동일한 부분이 핵심 클래스만 다룹니다.

### Encoder

```python
embedding_size = 256
hidden_size = 512
num_layers = 2
num_dirs = 2
dropout = 0.1
```

```python
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.gru = nn.GRU(
        input_size=embedding_size, 
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True if num_dirs > 1 else False,
        dropout=dropout
    )
    self.linear = nn.Linear(num_dirs * hidden_size, hidden_size)

  def forward(self, batch, batch_lens):  # batch: (B, S_L), batch_lens: (B)
    # d_w: word embedding size
    batch_emb = self.embedding(batch)  # (B, S_L, d_w)
    batch_emb = batch_emb.transpose(0, 1)  # (S_L, B, d_w)

    packed_input = pack_padded_sequence(batch_emb, batch_lens)

    h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers*num_dirs, B, d_h) = (4, B, d_h)
    packed_outputs, h_n = self.gru(packed_input, h_0)  # h_n: (4, B, d_h)
    outputs = pad_packed_sequence(packed_outputs)[0]  # outputs: (S_L, B, 2d_h)

    forward_hidden = h_n[-2, :, :]
    backward_hidden = h_n[-1, :, :]
    hidden = self.linear(torch.cat((forward_hidden, backward_hidden), dim=-1)).unsqueeze(0)  # (1, B, d_h)

    return outputs, hidden
```

* 3, 4강에 등장한 인코더와 동일하다. 다만, 내부 인자가 살짝 달라서 추가된 코드가 있다. 여기서는, layer의 수가 2개이고 방향도 양방향이다.
  * 그래서 hidden state의 3차원 개수가 1에서 4로 증가했다.
  * 또한, layer가 2개이므로 `forward_hidden` 을 첫번째 layer로, `backward_hidden` 을 두번째 layer로 정했고 실제 hidden state를 반환할 때는 이 둘은 `cat` 해서 반환했다.



디코더는 이전과 동일하므로 생략한다.



### Seq2Seq 모델 구축

```python
class Seq2seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2seq, self).__init__()

    self.encoder = encoder
    self.decoder = decoder

  def forward(self, src_batch, src_batch_lens, trg_batch, teacher_forcing_prob=0.5):
    # src_batch: (B, S_L), src_batch_lens: (B), trg_batch: (B, T_L)

    _, hidden = self.encoder(src_batch, src_batch_lens)  # hidden: (1, B, d_h)

    input_ids = trg_batch[:, 0]  # (B)
    batch_size = src_batch.shape[0]
    outputs = torch.zeros(trg_max_len, batch_size, vocab_size)  # (T_L, B, V)

    for t in range(1, trg_max_len):
      decoder_outputs, hidden = self.decoder(input_ids, hidden)  # decoder_outputs: (B, V), hidden: (1, B, d_h)

      outputs[t] = decoder_outputs
      _, top_ids = torch.max(decoder_outputs, dim=-1)  # top_ids: (B)

      input_ids = trg_batch[:, t] if random.random() > teacher_forcing_prob else top_ids

    return outputs
```

* encoder의 output은 사용하지 않는 모습.
* 또한, decoder의 output은 encoder처럼 한번에 나오지 않으므로 for문으로 작동시킨다. 그래서 이를 담아주기 위한 outputs를 선언해준다. 

