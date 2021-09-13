---
description: '210909, 210911'
---

# \(06강\) Beam Search and BLEU score

## 2. Beam search

자연어 생성 모델에서 Test 단계에서 좋은 품질의 생성 결과를 얻게하는 기법이다.

### Greedy decoding

우리는 현재 스텝에서 바로 다음단어를 예측하는데, 매 타임스텝마다 가장 높은 확률을 가지는 단어를 택해 디코딩을 하게된다. 이를 Greedy decoding이라고 한다. 전체 문장을 보고 다음단어를 예측하는 방식이 아닌, 매 타임스텝에서 다음 단어를 예상하는 방식이다.

만약 테스트 단계에서, 단어를 잘못 예측해서 다음 타임스텝부터 예측이 계속 틀리게 된다면 어떻게 해야될까?

### Exhaustive search

![](../../../.gitbook/assets/image%20%281109%29.png)

우리는 이 확률값이 높은 단어를 선택하게 된다. 이 확률은 joint distribution으로 나타내지는데, 현재 우리의 방식은 P\(y1\|x\)가 제일 높은 y1을 택하는 방식이다. 그러나, 전체적으로 봤을 때는 P\(y1\|x\)가 높더라도 y1을 택한 뒤의 확률들은 낮을 수 있고, P\(y2\|x\)가 낮더라도 y2를 택한 뒤에 확률들은 높을 수 있다.

그러나, 모든 경우의 수를 다 확인하려면 vocabulary size V와 time step T에 대하여 $$ V^T $$만큼의 경우의 수를 고려해야 하고, 이는 매우 높은 시간복잡도를 가지게 된다. 그래서 등장하는 차선책법이 Beam Search 이다.

### Beam search

핵심 아이디어는 디코더의 매 time step마다 우리가 정해놓은 k개의 가지수를 고려하는 것. 그리고 이 k개의 candidate 중에서 확률이 가장 높은 것으로 결정한다. k는 beam size라고 부르며 일반적으로 5~10의 값을 가진다.

![](../../../.gitbook/assets/image%20%281112%29.png)

* 계산을 작게 하기 위해 log를 취한다.

모든 경우의 수를 다 따지는 것은 아니기 때문에 가장 최적의 솔루션을 제공한다고 말할 수는 없지만, exhaustive search보다는 더 효율적인 방법이다.

![](../../../.gitbook/assets/image%20%281110%29.png)

* 확률값은 0~1의 값을 가지기 때문에 log 함수는 음수의 값을 가지게 되며 확률이 커질수록 0에 가까운 값을 가지게 된다.
* 이 때는 2의 제곱만큼 점점 가지수가 늘어나는 것이 아니라 항상 beam size만큼의 높은 확률을 가진 leaf에서만 다음 단어를 예측하게 된다. 아래 그림으로 이해할 수 있다.

![](../../../.gitbook/assets/image%20%281117%29.png)

![](../../../.gitbook/assets/image%20%281113%29.png)

* 생성을 끝나는 시점은 &lt;EOS&gt; 를 예측했을 때이다. beam search decoding 방식은 각각의 트리에서 서로 다른 시점에 &lt;EOS&gt; 토큰을 찾게되고 이러한 결과를 저장하게 된다.
* 그러면 빔 서치는 언제 끝날까? 다음과 같은 두 가지 방법이 있다.
  * T라는 최대 타임스텝을 정해서 이 시점까지만 디코딩하도록 한다
  * &lt;EOS&gt; 토큰을 만나 저장된 결과가 최소 n개 이상 있을 경우 멈추도록 한다
* 저장된 결과에서 가장 높은 결합 확률을 가진 가설을 최종 예측값으로 반환한다.
  * 이 때, 각 가설들은 끝나는 시점이 다르기 때문에, 상대적으로 긴 가설은 결합 확률이 낮고, 짧은 가설은 결합 확률이 높다. 
  * 공평하게 비교를 하기 위해서 hypotheses의 길이만큼으로 score를 나누어서 워드 당 평균 확률로 비교하게 된다.



## 3. BLUE score

![](../../../.gitbook/assets/image%20%281127%29.png)

위와 같은 상황에서 실제로 3개의 Ground Truth 중에 2개를 예측했지만 각각의 타임스텝에서 예측해야하는 값과는 다르기 때문에 정확도가 0%가 나오게 된다.

따라서, 단순히 정확도에 근거해서는 제대로 평가하지 못하게된다.

![](../../../.gitbook/assets/image%20%281124%29.png)

* 정밀도는 자신이 예측한 글자에서 몇개의 글자가 맞았는지에 대한 것
* 재현율은 실제 정답중에 내가 예측한 글자가 몇개 맞았는지에 대한 것

그리고 이 둘을 종합한 수치가 F-measure이다. 흔히, 종합을 할 때는 평균을 내게 되는데, 그 대표적인 평균 방법이 산술평균이다. 그 외에도 기하평균, 조화평균이 있다. 이 세가지 평균은 다음과 같은 특징이 있다.

> 산술평균 &gt;= 기하평균 &gt;= 조화평균

산술평균은 두 값에 대해 1:1로 내분되는 지점을 가리키며, 기하평균은 이것보다 좀 더 작은값에 대한 지점을, 조화평균은 더욱 더 작은값에 대한 지점을 가리키게된다.

결국, 이는 좀 더 작은값에 대해 가중치를 두는 방식이라고 볼 수 있고, 여기서 F-meausre는 조화평균을 사용한다.

그러나, 다음과 같은 예시에서 문제가 발생한다.

![](../../../.gitbook/assets/image%20%281123%29.png)

정답 단어를 모두 포함하나, 순서가 엉망진창인 문장에서도 100%의 점수를 내는 것이다. 이를 보완하기 위해서 BLEU Score가 고안된다.



### BLEU Score

개별 단어에 대해 Ground Truth에 포함되는 여부 뿐만 아니라 연속된 n개의 단어가 Ground Truth와 얼마나 겹치는가 까지 따지는게 BLEU Score의 특징이다.

이 때, Precision만을 고려하고 Recall은 무시하게 되는데, 이는 Precision의 특성과 관련이 있다. 다음과 같은 문장이 있다고 하자.

> Ground Truth : I love this movie very much.
>
> Prediction : 나는 이 영화를 많이 사랑한다.

실제로 이 예시에서는 very에 해당하는 '정말' 이라는 단어가 생략된 것을 알 수 있다. 그러나 그렇다 할지라도 번역된 문장이 높은 정확도를 가진 문장이라는 것을 알 수 있다.

그래서, 주어진 문장을 하나 하나 빠짐없이 변역했는가에 대한 Recall 보다는, 번역 결과만을 보고 주어진 문장에서 나오는 의미를 잘 표현했는가에 대한 Precision에 주목한다.

연속된 n개의 단어를 볼 때 이 n에 따라, Unigram, Bigram, Trigram, Fourgram, ... 으로 이야기 할 수 있다. BLEU는 이 4가지 경우를 모두 구하고 기하평균을 내서 구한다.

![](../../../.gitbook/assets/image%20%281129%29.png)

* 단순히 산술평균을 하지 않고, 기하평균을 함으로써 좀 더 작은 값에 가중치를 두겠다는 의도를 가지고 있다.
* 또, 조화평균을 쓰지 않은 이유는 다른 평균방법들과 비교적 작은 값에 지나친 가중치를 두는 경향이 있어서이다.

또, BLEU는 brevity penalty를 사용하는데 만약에 reference의 길이보다 짧은 문장을 예측했을 때, 짧아진 비율만큼 precision값을 낮춰주고, 긴 문장을 예측했을 때는 재현율의 최대로 마지노선을 설정했다. 그래서 Recall을 아예 고려를 하지않은 것은 아니다라고 할 수 있다.

* 만약 정답 문장이 10 단어이고, 예측 문장은 7단어이면 0.7의 값을 갖게 되며 10단어이고 예측 문장은 15단어 라면 1.5의 값을 갖게되지만 1의 값으로 제한해준다.

![](../../../.gitbook/assets/image%20%281128%29.png)

* 1-gram : 각 단어가 Ground Truth에 포함되어 있는지 확인
* 2-gram : 연속된 두 단어가 GT에 포함되어 있는지 확인
* 이하 동일
* 이후, 모든 Precision의 값을 곱하고 4중근의 형태로 만든 뒤 이를 Brevity penalty와 곱해서 점수를 얻게된다.



## 실습

5강의 실습과 동일하고, 여기서는 추가적으로 attention 부분이 추가되었다.

### **Dot-product Attention 구현**

```python
class DotAttention(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, decoder_hidden, encoder_outputs):  # (1, B, d_h), (S_L, B, d_h)
    query = decoder_hidden.squeeze(0)  # (B, d_h)
    key = encoder_outputs.transpose(0, 1)  # (B, S_L, d_h)

    energy = torch.sum(torch.mul(key, query.unsqueeze(1)), dim=-1)  # (B, S_L)

    attn_scores = F.softmax(energy, dim=-1)  # (B, S_L)
    attn_values = torch.sum(torch.mul(encoder_outputs.transpose(0, 1), attn_scores.unsqueeze(2)), dim=1)  # (B, d_h)

    return attn_values, attn_scores
```

* 인코더의 output과 디코더의 hidden state를 가지고 내적을 하는 모습. 이렇게 구한 내적을 softmax를 거쳐서 최종값을 구한뒤 각 임베딩 벡터의 가중치로 생각하고 마지막으로 attention output을 구하는 모습이다.

















