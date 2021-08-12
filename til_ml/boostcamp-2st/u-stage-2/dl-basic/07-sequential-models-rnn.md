---
description: '210812'
---

# \(07강\) Sequential Models - RNN

## Sequential Model

일상 속에 있는 대부분의 데이터\(동작이나 모션, 영상, 음성\)이 Sequential이다. 이런 데이터의 특징은 입력 크기를 알 수가 없다는 것이다. 그래서 이전에 사용한 FC나 CNN을 사용하기 어렵다.

### Naive sequence model

이전 데이터를 가지고 다음 데이터를 예측하는 것

![](../../../../.gitbook/assets/image%20%28848%29.png)

### Autoregressive model

그에 반해 Auto. 모델은 이전 n개 데이터를 가지고 다음 데이터를 예측한다. n개로 이전 데이터의 개수가 고정되므로 계산이 쉬워진다.

![](../../../../.gitbook/assets/image%20%28863%29.png)

* 미래는 과거 n개에만 independent하다라는 가정

### Markov model

Markov Assumption을 가지기 때문에 다음과 같은 이름이 붙음.

* 강화학습을 공부할 때 나오기도 한다. Markov Decision Process, MDP

말이 되는 논리는 아니다. 예를 들어 내일 수능은 오늘 공부한 것만으로 결정된다는 논리. 그래서 이러한 markov 모델은 많은 정보를 버리게 된다. 장점은 joint distribution을 설명할 때 되게 좋아진다.

### Latent autogressive model

이전 모델의 가장 큰 단점은 과거의 많은 데이터를 고려해야 하는데 그럴 수 없었다. 이 모델은 과거의 정보를 hidden state가 요약한다는 것. 그리고 다음 time step은 이 hidden state 하나에만 의존한다.

![](../../../../.gitbook/assets/image%20%28841%29.png)

## Recurrent Neural Network

앞에서 보았던 모델들과 다 동일하지만 한 가지의 차이점은 RNN은 자기 자신으로 돌아오는 구조가 있다는 것

![](../../../../.gitbook/assets/image%20%28861%29.png)

RNN을 시간순으로 데이터를 풀게되면 사실 입력이 굉장히 많은 FC로 볼 수 있다.

RNN의 큰 문제는 `Short-term dependencies` 과거의 정보들이 계속 추합되기 때문에 시간이 지날수록 과거의 정보들이 현재의 상태\(그림에서 A\)에 유지되기 어렵다는 것.

* 예를 들어, 음성인식의 경우 5초 까지는 잘 기억하는데, 5초 이전에 말했던 음성들은 제대로 반영이 안되는 문제

![](../../../../.gitbook/assets/image%20%28860%29.png)

h0가 h4까지 가기 위해 여러 연산을 거쳐야 하게 된다. 만약 activation 함수가 sigmoid라면 이 h0값이 squshing되게 되고 이를 반복하다 보니 남은 값의 의미가 없어지게 된다

* squshing된다라는 것은 찌부러지다 라는 의미인데, 입력 값이 아무리 커도 sigmoid를 거치면 0~1로 줄어든다는 것을 의미

반대로, activation 함수가 ReLU라면 h0값이 h4까지 갈 때마다 반복적으로 곱해지고 결국 h4에 도달했을 때 h0의 값이 매우 크게 된다.

정리하면, sigmoid는 vanishing gradient 문제가, ReLU는 exploding gradient 문제가 발생하게 된다.

* 그래서 RNN을 할 때 ReLU를 쓰지 않는다.
* 그래서 LSTM이라는 새로운 모델이 나오게 되었다.

## Long Short Term Memory

기본적인 RNN\(Vanilla\)의 구조는 다음과 같다

![](../../../../.gitbook/assets/image%20%28852%29.png)

LSTM의 구조는 다음과 같다

![](../../../../.gitbook/assets/image%20%28867%29.png)

![](../../../../.gitbook/assets/image%20%28844%29.png)

* X는 입력, H는 출력이다. 
* cell state는 셀 내부를 관통하며 밖으로 흐르지는 않는다.
  * previous cell state는 t-1 개의 정보를 취합해서 요약해주는 역할
* 잘 보면, output이 밖으로 나가기도 하고 next hidden state로 빠지기도 한다. 이 것이 전단계에서 오면 previous hidden state
* 그리스어로 생긴 모양은 모두 sigmoid 함수를 의미한다.

LSTM을 이해할 때는 gate 위주로 이해하면 좋다. LSTM은 3개의 gate로 이루어져 있는데 forget gate, input gate, output gate 이다.

LSTM의 가장 큰 아이디어는 중간에 흘러가는 Cell state이다

### Forget Gate

![](../../../../.gitbook/assets/image%20%28845%29.png)

어떤 정보를 버릴지\(잊어버릴지\) 결정한다. 이전의 셀 정보와 현재 셀의 입력이 들어가서 $$ f_t$$ 라는 숫자를 얻는다. 이 f는 sigmoid의 값이므로 항상 0에서 1사이의 값을 갖는다

### Input Gate

![](../../../../.gitbook/assets/image%20%28865%29.png)

현재 입력을 무작정 cell state에 올리는 것이 아니라 올릴 정보를 결정한다. 입력과 이전 셀 정보를 이용해서 $$ i_t $$ 라는 정보를 만들게된다. 이제 이 i에서 올릴 정보를 결정해야 하는데, 이는 $$ C_t $$ 로 결정하게 된다. \(읽는 법은 C TILDE\) 이 C 역시 이전의 셀 정보와 현재 입력값을 가지고 다른 학습된 NN을 통해서 얻는 값이다. 이 값은 hyperbolic tangent를 통과하기 때문에 -1에서 1사이의 값을 가진다.

### Update Cell

![](../../../../.gitbook/assets/image%20%28858%29.png)

Forget gate에서 구한 f와 Input gate에서 구한 i를 가지고 현재셀의 cell state를 갱신한다.

### Output Gate

그대로 Cell State를 Output으로 뽑을 수도 있다. 이것을 GRU에서 한다

* _Gated Recurrent Units, GRU_는 RNN 프레임워크의 일종으로 LSTM보다 더 간랸한 구조이다

LSTM에서는 이것을 한번더 변환한다. 얼마나 바깥으로 빼낼지를 결정하는 부분

## Gated Recurrent Unit

일반적으로 RNN을 쓰면 Vanilla RNN, LSTM 그리고 GRU를 쓴다.

![](../../../../.gitbook/assets/image%20%28862%29.png)

GRU는 게이트가 2개이다.

* Reset Gate, Update Gate

히든 스테이트가 곧 아웃풋이고 바로 다음 스테이트로 들어가게 된다.

* RNN은 아웃풋을 변환했다.









