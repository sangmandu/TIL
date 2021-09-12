---
description: '210826'
---

# \(8강\) Training & Inference 2

### Training Process

#### model.train\(\)

`model.train(True)` 와 동일하다. 모델을 학습하겠다라는 뜻. 이 부분을 해주지 않으면 학습할 때 있어서 Batch Normalization과 Dropout이 적용되지 않는다. 이러한 테크닉은 CNN에 있어서 거의 필수적으로 포함되는 기술이기 때문에 이것을 적용하려면 이 부분을 선언해줘야 한다.



#### optimizer.zero\_grad\(\)

이를 해주지 않으면 이전 배치에서 계산된 grad에 현재 배치에서 계산된 gread가 더해지게된다.



#### loss = criterion\(outpus, labels\)

loss를 컨트롤 하면서 전체적인 파라미터를 컨트롤할 수 있게된다.

![](../../../.gitbook/assets/image%20%281013%29.png)

계속적으로 next\_function이 연결되어 체인구성으로 이루어져있는 모습. loss.backward\(\)가 이루어지면 모든 파라미터의 grad 값이 갱신이 된다. 그러나 실질적으로 파라미터가 갱신된 것은 아니다. 파라미터를 갱신하려면 다음을 실행해야한다.



#### optimizer.step\(\)

이 함수를 거치면, 파라미터의 grad값을 가지고 파라미터값을 갱신하게된다.



### Gradient Accumulation

만약에 배치 사이즈를 크게 하고 싶은데, GPU 리소스가 부족하다면 어떻게 할까? 어쩔 수 없이 작은 배치 사이즈로 돌릴 것이다. 이 때 매번 작게 설정한 배치마다 optimizer.step\(\) 이 이루어지게 되는데, 



### Inference Process

#### model.eval\(\)

`model.train(False)` 와 동일하다. 모델을 평가하겠다는 뜻. 여기서는 드랍아웃이나 배치정규화 기능이 꺼지게된다.



#### with torch.no\_grad\(\)

이 영역에서부터 모든 파라미터의 grad는 False값을 가진다는 것이다.



#### Validation

추론 과정에 Validation 셋이 들어가면 이것이 검증과정이다. Test 셋과 큰 차이점은 없다.



#### Checkpoint

보통 Validation셋의 성능을 보고 모델을 저장할지 말지 결정하게 된다. 왜냐하면 Validation 데이터는 Model에 feed되지 않았기 때문이다.





