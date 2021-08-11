---
description: '210811'
---

# \(2-1\) Bar Plot 사용하기

## Bar Plot

직사각형 막대를 사용하여 데이터의 값을 표현하는 차트/그래프

* 막대 그래프, bar chart, bar graph의 여러 이름을 가짐
* 범주에 따른 수치 값을 비교하기에 적합
  * 개별비교, 그룹비교 모두 적합하다



막대의 방향에 따른 분류

![](../../../.gitbook/assets/image%20%28823%29.png)

* 수직 : `.bar()`
  * 기본적으로 사용한다
* 수평 :  `.barh()`
  * 범주가 많을 때 사용한다



## 다양한 Bar Plot

```python
Sky = [1, 2, 3, 4, 3]
Pink = [4, 3, 2, 5, 1]
```

위 두 데이터를 비교하기 위한 여러 방법을 사용할 것임



### Multiple Bar Plot

1. 플롯을 여러 개 그리는 방법
2. 한 개의 플롯에 동시에 나타내는 방법
   * 쌓아서 표현
   * 겹쳐서 표현
   * 이웃에 배치하여 표현



### Stacked Bar Plot

![](../../../.gitbook/assets/image%20%28824%29.png)

2개 이상의 그룹을 쌓아서 표현하는 bar plot

* 이 때 각 bar에서 나타나는 그룹의 순서는 유지해야 한다. 그렇지 않으면 혼동을 줄 수 있다.

맨 밑에 bar의 분포는 파악하기 쉽다

* 그러나 그 외의 분포들은 파악하기 어렵다
  * sky 데이터는 파악하기 쉽지만, pink 데이터는 파악하기 어렵다.
  * 이럴 때는 수치를 annotation 달 것을 추천한다.
* 2개의 그룹이 positive/negative 라면 축 조정이 가능하다
* .`bar()` 에서는 bottom 파라미터를 사용하고 `.barh()` 에서는 left 파라미터를 사용한다

좀 더 데이터 분포 비교가 원활한 `Percentage Stacked Bar Chart` 도 있다.

![](../../../.gitbook/assets/image%20%28822%29.png)

* 각 bar 마다 퍼센트 수치를 알려주는 annotation이 표기되어 있는 모습



### Overlapped Bar Plot

![](../../../.gitbook/assets/image%20%28825%29.png)

2개 그룹만 비교한다면 겹쳐서 만들 수도 있다.

* 3개 이상부터는 파악이 어렵기 때문

같은 축을 사용하기 때문에 비교와 구현이 쉽 투명도를 조정해서 겹치는 부분을 파악해야 한다

Bar Plot보다는 Area Plot에서 더 효과적이다.



### Grouped Bar Plot

그룹별 범주에 따른 bar를 이웃되게 배치하는 방법이다.







