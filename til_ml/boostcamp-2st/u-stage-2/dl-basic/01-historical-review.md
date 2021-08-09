---
description: '210809'
---

# \(01강\) 딥러닝 기본 용어 설명 - Historical Review

## Introduction

딥러닝이라는 학문은 너무 큰 분야라 모든 분야를 다 커버할 수가 없다.

![](../../../../.gitbook/assets/image%20%28789%29.png)

우리 수업에서는, 10번에 걸쳐서 꼭 알아야 하는 내용들만을 다룸

* 교수님이 생각하기에 필수적이라고 생각하는 부분



### 개념

![](../../../../.gitbook/assets/image%20%28787%29.png)

인공지능

* 인간의 지능을 모방하는 것
* 가장 큰 개념

머신러닝

* 인공지능 안에 있는 분야
* 데이터로 학습을 해서 알고리즘을 구현한 모델

딥러닝

* 머신러닝 안에 있는 분야
* 신경망을 사용하는 모델



### 딥러닝의 중요요소

* 학습할 데이터
* 데이터를 학습할 모델
* 학습할 모델의 비용함수
* 인자를 조정할 알고리즘

연구를 볼 때 이 4가지를 중점적으로 보면 연구를 잘 이해할 수 있다.



### Data

데이터는 해결해야할 문제의 유형에 따라 결정된다

* Classification : 이미지를 분류
* Semantic Segmentation : 단순히 이미지를 분류할 뿐만 아니라 각 픽셀별로 어떠한 클래스에 속하는지 분류한다
* Detection : 이미지 내에 클래스에 대해 바운딩박스를 찾는 것
* Pose Estimation : 이미지 내에 2차원 정보 또는 3차원 정보를 찾는 것
* Visual QnA : 이미지를 참고하여 질문이 주어졌을 때 답을 도출해 냈는 것



### Model

같은 데이터가 주어지고 같은 문제가 주어졌더라도 모델의 종류에 따라 성능이 달라진다.



### Loss

기준이 되는 비용함수를 정해야 한다

![](../../../../.gitbook/assets/image%20%28786%29.png)

그러나, 각 문제별로 비용함수가 정해져 있는 것은 아니다. 회귀문제를 풀 때 노이즈가 많은 경우에는 에러가 높아지기 때문에 L1 Norm을 사용하는 등의 여러 방법이 있다.

중요한 것은 우리가 풀고자 하는 문제에 대해 Loss 함수가 줄어드는 것이 어떻게 작용되는 지, 그리고 왜 사용하는 지가 중요하다 



### Optimization Algorithm

![](../../../../.gitbook/assets/image%20%28788%29.png)

신경망의 파라미터를 1차 미분한 정보를 활용한다.

* SGD : 그냥 활용한다
* 그 이외 방법 : 여러 인자들을 곱해서 활용한다

그 외에도 규제화, 드랍 아웃 등을 통해 테스트 데이터에 대한 성능을 높이는 방법들이 있다



## Historical Review

### 2012 - AlexNet

* 딥러닝을 이용해서 처음으로 수상한 모델



### 2013 - DQN

* 강화학습방법을 딥러닝에 적용한 것
* 알파고의 시작
* 오늘날의 딥마인드가 있게한 논문



 


