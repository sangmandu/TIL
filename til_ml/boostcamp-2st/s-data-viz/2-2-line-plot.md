---
description: '210814'
---

# \(2-2\) Line Plot 사용하기

## 1. 기본 Line plot

Line Plot

* 연속적으로 변화하는 값을 순서대로 점으로 나타내고, 이를 선으로 연결한 그래프
* 시간이나 순서에 대한 변화를 볼 수 있고 추세를 살피기 위해 사용한다
  * 시계열 분석에 특화
* `.line` 이 아니라 `.plot()` 을 사용한다.
* 5개 이하의 선을 사용하는 것을 추천한다
  * 선이 많으면 가독성이 하락한다
* 선을 구별하는 요소는 다음과 같다
  * 색상
  * 마커 : 점을 동그라미, 세모 등으로 표현하기
  * 선의 종류 
* 시시각각 변동하는 데이터는 노이즈가 커 패턴이나 추세를 파악하기 어렵기 때문에 노이즈를 줄이기 위해 스무딩을 사용한다

## 2. 정확한 Line Plot

### 2.1 추세에 집중

Bar plot과 다르게 꼭 축을 0에 초점을 둘 필요가 없다. 또 구체적인 line plot 보다는 생략된것이 나을 수 있다.

![](../../../.gitbook/assets/image%20%28934%29.png)

깔끔한 그래프를 원한다면 항상 후자가 좋아보이지만 정확한 데이터 수치가 중요하다면 왼쪽도 좋은 그래프이다

### 2.2 간격


