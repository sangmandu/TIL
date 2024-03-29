---
description: '210805'
---

# \(AI Math 10강\) RNN 첫걸음

### 시퀀스 데이터

소리, 문자열, 주가, 비디오 등의 연속적인 데이터를 시퀀스 데이터라고 한다.

시퀀스 데이터는 독립동등분포 가정을 잘 위배하기 때문에 순서를 바꾸거나 과거 정보에 손실이 발생하면 데이터의 확률분포도 바뀌게 된다.

* 독립동등분포는 각각의 시행이 독립적으로 시행되는 확률분포를 의미한다.
  * 주사위를 굴려 6이 나오는 사건과 주사위를 굴려 1이 나오는 사건은 독립이다
  * 내가 지나가는 사람을 때리는 사건과 지나가는 사람이 나를 쫓아 뛰는 사건은 독립이 아니다.



시퀀스 데이터는 조건부 확률을 이용할 수 있다.

![](../../../../.gitbook/assets/image%20%28756%29.png)

* 왜냐하면 사건간의 종속성이 있기 때문이다.
  * 이 때 베이즈 정리를 이용한다.
* 시퀀스 데이터를 다루기 위해서는 길이가 가변적인 데이터를 다룰 수 있는 모델이 필요하다



### Recurrent Neural Network

가장 기본적인 RNN 모형은 MLP와 유사한 모양이다.

![](../../../../.gitbook/assets/image%20%28783%29.png)

입력 벡터 $$X_t$$와 이전 잠재변수 $$ H_{t-1} $$를 이용하여 현재 셀의 잠재변수 $$ H_t $$ 를 만들고 이를 이용해 결과물인 $$ O_t$$를 반환한다.

이 때 가중치는 3가지가 필요하다

* 입력벡터 $$ X $$와 곱해지는 $$ W_X^{(1)}$$
* 잠재변수 $$ H $$와 곱해져서 새로운 잠재변수를 만드는 $$W_H$$
* 잠재변수 $$ H$$와 곱해져서 출력을 만드는 $$W^{(2)}$$

이 때 W는 레이어의 층 t 에 따라 변하지 않는 고정적인 가중치이다.



RNN의 역전파는 잠재변수의 연결그래프에 따라 순차적으로 계산한다. 이 방법을 Backpropagtaion Through Time, BPTT 라 하며 RNN의 역전파 방법이다.



### BPTT

BPTT를 통해 RNN의 가중치행렬의 미분을 계산하면 아래와 같이 미분의 곱으로 이루어진 항이 된다.

![](../../../../.gitbook/assets/image%20%28782%29.png)











