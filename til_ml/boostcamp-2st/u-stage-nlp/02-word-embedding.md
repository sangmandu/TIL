---
description: '210906'
---

# \(02강\) Word Embedding

## 1. Word Embedding : Word2Vec, GloVe

워드 임베딩은 자연어가 단어들을 정보의 기본 단위로 해서 각 단어들을 특정 공간에 한점으로 나타내는 벡터로 변환해주는 기법이다.

고양이를 의미하는 cat과 어린 고양이를 의미하는 kitty는 의미가 유사하므로 각 점은 가까이 위치하고 hamburger와는 멀리 위치하게 된다.

### Word2Vec

워드 임베딩을 하는 방법 중 대표적인 방법. 같은 문장에서 나타난 인접한 단어들 간에 의미가 비슷할 것이라는 가정을 사용한다. "The cat purrs" 와 " This cat hunts mice" 라는 문장에서 cat이라는 단어는 The, purrs, This, hunts, mice 와 관련이 있다.

![](../../../.gitbook/assets/image%20%281051%29.png)

어떠한 단어가 주변의 등장하는 단어를 통해 그 의미를 알 수 있다는 사실에 착안한다. 주어진 학습 데이터를 바탕으로 cat 주변에 나타나는 주변 단어들의 확률 분포를 예측하게 된다. 보다 구체적으로는 cat을 입력단어로 주고 주변단어를 숨긴채 예측하도록 하는 방식으로 Word2Vec의 학습이 진행된다.



구체적인 학습 방법은 다음과 같다.

* 처음에는 "I study math" 라는 문장이 주어진다
* word별로 tokenization이 이루어지고 유의미한 단어를 선별해 사전을 구축한다
* 이 후 사전에 있는 단어들은 사전의 사이즈만큼의 차원을 가진 ont-hot vector로 표현된다.
* 이후 sliding window라는 기법을 적용해서 한 단어를 중심으로 앞뒤로 나타나는 단어들 각각과의 입출력 단어 쌍을 구성하게된다.
  * 예를 들어 window size가 3이면 앞뒤로 하나의 단어만을 보게된다.
  * 중심 단어가 I 라면 \(I, study\) 라는 쌍을 구할 수 있게 된다.
  * 중심 단어가 study 라면 \(study, I\) 와 \(study, math\) 라는 쌍을 얻을 수 있다.
  * 즉, \(중심 단어, 주변 단어\) 라는 관계를 가진 쌍을 window size에 따라 만들어 낼 수 있게된다.

![](../../../.gitbook/assets/image%20%281043%29.png)

* 이렇게 만들어진 입출력 단어 쌍들에 대해 예측 Task를 수행하는 Two layer를 만들게 된다.
  * 입력과 출력노드의 개수는 Vocab의 사이즈와 같다.
  * 가운데에 있는 Hidden layer의 노드 수는 사용자가 정하는 하이퍼 파라미터이며, 워드임베딩을 수행하는 차원 수와 동일한 값으로 주로 결정한다.

![](../../../.gitbook/assets/image%20%281048%29.png)

* 만약 \(study, math\) 쌍을 학습한다고 하자. 그러면 input값으로 study를 의미하는 \[0, 1, 0\] 이 입력된다.
  * study : \[0, 1, 0\]
  * math : \[0, 0, 1\]
  * I : \[1, 0, 0\]
* Input layer는 3차원 Hidden layer는 2차원이므로 W는 3 \* 2의 형태를 가져야 하며 실제 X와 곱해질 때의 2 \* 3의 모양으로 곱해진다.
* Output layer는 3차원이므로 W는 3 \* 2의 모양으로 곱해진다.
* 이후, Softmax를 적용해서 확률분포 벡터를 얻게된다. 그리고 이를 Ground Truth와의 거리가 제일 가까워지게 하는 Softmax Loss를 적용함으로써 W1과 W2를 학습하게된다.
* 여기서 W1과 X를 내적하게 되는데, X의 특성상 특정 인덱스만 1이고 나머지는 다 0이다 보니, W1의 특정 인덱스 값만 뽑아온다고 볼 수 있다.
  * 그래서 코드로 구현할 때에도 내적을 구현하지 않고 X에서 1이 존재하는 인덱스를 가지고 W1에서 가져오게된다.
* 마찬가지로 W2와 W1\*X를 내적할 때도 Ground Truth값인 Y에서 1이 존재하는 인덱스에 해당하는 값만 확인하면 된다. 따라서 W2는 Y에서 1이 존재하는 인덱스에서의 값만 뽑아온다.
  * 예측값과 실제값이 가까워지려면 W2 \* W1 \* X의 값은 정답에 해당하는 인덱스의 값은 무한대에 가까워야 하고 그외의 값들은 음의 무한대에 가까워야 한다.
  * 그래야 Softmax를 적용했을 때 양의 무한대에 대해서만 1을 얻고 나머지 위치에서는 0을 얻기 때문

{% embed url="https://ronxin.github.io/wevi/" %}

위 링크에 들어가면 워드 임베딩을 시각적인 상태로 볼 수 있다. 예시로는 8개의 단어를 사용했으며 hidden size는 5이다.

![](../../../.gitbook/assets/image%20%281044%29.png)

또, 다음과 같이 W1과 W2 행렬을 시각적으로 확인할 수 있다.

* W1은 Transpose를 통해 W2와 사이즈가 같도록 나타냈다.
* 푸른색은 음수, 붉은색은 양수이다.
* 현재는 Random Initialization 된 상태이다.

![](../../../.gitbook/assets/image%20%281036%29.png)

각 단어의 임베딩 된 좌표평면도 다음과 같다.

* W1과 W2의 차원은 5개지만 PCA를 통해서 2-Dimension으로 차원 축소를 한 뒤 Scatter plot의 형태로 각각의 벡터들을 시각화 한 결과이다.

![](../../../.gitbook/assets/image%20%281040%29.png)

Trainin data는 다음과 같다.

```text
eat|apple
eat|orange
eat|rice
drink|juice
drink|milk
drink|water
orange|juice
apple|juice
rice|milk
milk|drink
water|drink
juice|drink
```



이후, 300번의 학습을 진행하게 된 후의 결과를 확인해보자.

![](../../../.gitbook/assets/image%20%281039%29.png)

* juice의 input vector는 drink의 output vector와는 유사한 벡터를 가지게 되는 것을 볼 수 있다.
  * 이 두 단어벡터의 내적값은 커지게 된다.
  * 또한 milk와 water와도 유사하다.
* eat과 apple도 유사한 벡터를 가진다.
  * orange와도 유사하다.

입력 단어와 출력 단어의 두 개의 벡터를 최종적으로 얻을 수 있고 둘 중에 어느것을 워드 임베딩의 아웃풋으로 사용해도 상관이 없으나 통상적으로 입력 단어의 벡터를 사용하게 된다.

![](../../../.gitbook/assets/image%20%281045%29.png)

이렇게 학습된 Word2Vec은 Words간의 의미론적 관계를 Vector Embedding로 잘 표현할 수 있다.

다음 그림은 Word2Vec으로 학습된 단어들의 임베딩 벡터를 표현한 것이다.

![](../../../.gitbook/assets/image%20%281041%29.png)







