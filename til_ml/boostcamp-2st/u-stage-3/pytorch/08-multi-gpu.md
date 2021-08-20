---
description: '210820'
---

# \(08강\) Multi-GPU 학습

궁극적으로는 Multi Node Multi GPU를 하려고 하지만, 많은 어려움이 있어 현재는 Multi GPU 라고 함은 Single Node Multi GPU이다.



### Model parallel

다중 GPU에 학습을 분산하는 방법이다. 이 때 두가지로 나눌 수 있다.

* 데이터를 나누기
* 모델을 나누기
  * 예로 알렉스넷이 있다.

![](../../../../.gitbook/assets/image%20%28943%29.png)

모델 병렬은 고난이도 과제이며 병목등의 어려움이 발생할 수 있다.

![](../../../../.gitbook/assets/image%20%28940%29.png)

* 첫번째 파이프라인은 병렬이 되지 않는 모습. 두번째 파이프라인처럼 되어야 한다



### Data parallel

파이토치에서는 두 가지 방식을 제공한다

* DataParallel
  * 단순히 데이터를 분배한 후 평균을 취한다
  * 분배된 데이터를 받아서 계산하는 GPU가 있고, 계산한 결과를 종합하는 GPU가 있는데, 종합하는 GPU가 상대적으로 메모리를 많이 쓰게 돼서 GPU간의 불균형 문제가 발생한다. 그래서 병목현상이 일어난다.
  * 이는 각 GPU에서 처리하는 배치사이즈를 감소시키는 해결방안이 있다
* DistrubutedDataParallel
  * 각각의 GPU가 계산해서 종합하는 방법이다. 이 때에는 GPU수만큼 CPU가 존재한다.





