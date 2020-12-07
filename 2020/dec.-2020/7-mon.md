---
description: TIL
---

# 7 Mon

## 프로그래머스 AI 스쿨 1기

#### 2주차 DAY 1

### 쥬피터 실습

![](../../.gitbook/assets/image%20%2836%29.png)

### Numpy 실

![](../../.gitbook/assets/image%20%2837%29.png)

![](../../.gitbook/assets/image%20%2831%29.png)

### 선형 시스템 대수적 표

![](../../.gitbook/assets/image%20%2839%29.png)

### 선형 시스템 실습 

![](../../.gitbook/assets/image%20%2829%29.png)

![](../../.gitbook/assets/image%20%2823%29.png)

![](../../.gitbook/assets/image%20%2838%29.png)

### 가우스 소거법 : Forward Elimination

![](../../.gitbook/assets/image%20%2830%29.png)

![](../../.gitbook/assets/image%20%2828%29.png)

### 가우스 소거법의 가치

![](../../.gitbook/assets/image%20%2824%29.png)

![](../../.gitbook/assets/image%20%2826%29.png)

![](../../.gitbook/assets/image%20%2827%29.png)

### LU 분해

약분이나 최대/최소 공배수를 구할 때 식이 인수분해가 가능하면, 더 쉽게 구할 수 있다.

마찬가지로,  행렬도 행렬 분해가 가능하면 쉽게 계산이 가능하다.

![](../../.gitbook/assets/image%20%2840%29.png)

y 구하기는 전방대치법으로, x 구하기는 후방대치법으로 가능하다.

![](../../.gitbook/assets/image%20%2832%29.png)

LU 분해는 가우스 소거법의 전방 소거법을 행렬로 코드화 한 것이다.

replacement : 행을 대치\(교체\)

scaling : 기준이 되는 부분을 1로 고정

이 때, P에는 interchange record가 저장된다. 실제로 Numpy LU 분해 리턴값은 P, L, U로 리턴된다.

![](../../.gitbook/assets/image%20%2834%29.png)

A의 역행렬을 구할 때 1\) 수치적으로 불안하다 2\) 반복적인 b의 변화에 대한 x 구하기는 PLU로 분해해 두면 빠르게 구할 수 있다 의 이유로 역행렬보다 LU 분해를 사용한다.



 

