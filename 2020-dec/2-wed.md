---
description: TIL
---

# 2 Wed

## 프로그래머스 AI 스쿨 1기

#### Day 2

오늘 안에 Day2와 Day3를 끝내겠다 다짐했건만, Day2를 다 풀지 못했다. 12시까지 해야 출석에 반영된다고 했는데 큰일난 것 같다. 자정을 넘겨서라도 얼른 해야겠다. 

![](../.gitbook/assets/image%20%289%29.png)

![](../.gitbook/assets/image%20%285%29.png)

![](../.gitbook/assets/image%20%283%29.png)

**Day 1** 에서는 너무 기초적인 게 아닌가 싶었는데, Day 2는 좀 더 유익했다.\(혹시, 내 수준은 여기서 부터인 걸까?\) 중위표현의 후위표현 부터 쉽지 않았던 것 같다. 확실히 처음 배울 때도 그랬지만, 이해 하는 것과 구현하는 것은 정말 다른 것 같다. 이럴 기회가 실제 대학 강의 이외에는 없었는데, 이렇게 다시 구현해보니까 큰 복습이 되는 것 같다. 생각보다 어려워서 실습 하나 풀기가 쉽지 않다. 다행히도 모든 자료구조와 동작 방식을 간단하게라도 알고 있어서 흡수는 바로바로 했다. 몇 개 나름 중요하다고 생각한 내

* 우선순위 큐는 삭제 할 때마다 해당 데이터를 찾는 것보다 삽입 할 때마다 데이터를 정렬하는 것이 더 빠르다.
* 우선순위 큐는 선형 배열로 구현할 경우 메모리에서, 연결 리스트로 구현할 경우 속도에서 강점을 얻는다.

#### Day 3

![](../.gitbook/assets/image%20%288%29.png)

dictionary의 초기화에 대해서 d.setdefault\(x, 0\)만 알고 있었는데, d.get\(x, 0\)의 문법을 알게 되었다. 둘의 차이점을 좀 더 자세히 알아보기 위해 다음 링크 참

{% embed url="https://stackoverflow.com/questions/7423428/python-dict-get-vs-setdefault" %}

get보다 setdefault의 속도가 10% 더 빠르다. 다만, setdefault는 unset된 dictionary에 대해서 초기화를 하는 것이고 get은 unset을 구분짓지 않고 reset을 할 수 있다는 특징이 있다. 

전반적으로 유료 강의에 대해 느낀 점은, 어떻게 풀어야 할지를 잘 설명하고 문제 풀이를 시작하는 강의라고 생각한다. 실제 프로그래머스에서 문제를 풀면 다른 사람의 코드를 확인할 수 있는데, 풀이 방법이 추천을 많이 받은 풀이에 대부분 존재한다. 그 사람들이 유료강의를 보고 풀은 것인지, 이 사람들의 풀이를 보고 유료강의 제작에 참고한 것인지는 모르겠다.\(아마 후자가 아닐까\). 다만 알려준 풀이 방법보다 더 좋은 방법도 몇 개 있으니 강의를 보는 사람들은 더 개선할 점을 생각해보면 좋을 것 같다.

또, AI 스쿨 1주차에 대해 느낀 점은 이미 CS 지식에 대해 어느 정도 알고 있으면 복습이 되고 보충이 될 것 같다. 그러나 최근에 프로그래머스의 여러 문제를 풀이했던 사람들은 문제 풀이에 대해서는 좀 시간 채우기 느낌이 들 것 같다. \(내가 후자의 느낌이라...\) 피드백이 있다면, 아직 AI를 배우기 전이지만 이러한 문법은 어떤 것을 배우는 데 필요하다 같은 적용 분야를 언급하거나 간단한 AI 지식을 적용하면서 이렇게 사용할 수 있다 라는 방식으로 배우면 좀 더 흥미로울 것 같다.

## 오토마타와 컴파일러

#### 실수 판별 토큰 분석

수강 중인 강의에 실수를 판별하는 오토마타의 DFA를 그리고 실제로 프로그래밍 하는 과제가 나왔다. 정규식으로 빠방 하니까 5줄만에 끝났는데, 토큰을 분석하는 느낌이 아니어서\(물론 정규식 내부에서는 토큰으로 분석하겠지만 내가 토큰마다 분석한 건 아니니까\) 여러 분기점을 만들었다.

![](../.gitbook/assets/image%20%284%29.png)

```python
#
#   data : 2020.12.02
#   author : sangmandu at Dankook Univ.
#   program : automata that analyze token by token and distinguish type of input is float or not
#


#
# prerequisite : the number could not be calculated by other operators or operand
# although there are many operators and operand and result of calculation number is regarded as certain type,
# here's token analyzer regards input as calculated number completely
# ex) -11-3.e-3 is float but token analyzer says this is no float as duplicate of '-'
#


# Test cases
X = ['+5-5', '5*5*5', '5-5-5', '1...5', '1.2.3', '1.5e15', '000.5', '0000.', '000100.5', '3.5', '2', '4.', '-5.3E+2', '36', '-52', '-13.E+3', '54.123E-2',
     '0', '0.0', '-0', '-0.0', '.35', '+++++3.5', '-----3.5','11.e+++3', '11.e---3', '11.e-3-', 'abc', '+35a', '-35b', '+35.5a', '123a.123', '23.12c', '11.232e++']
Y = [False, False, False, False, False, True, True, True, True, True, False, True, True, False, False, True, True,
     False, True, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False]
P = []

#
# Float Regular Expression
# Re : [+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?
#
# 0 : [+-]      1 : [0-9]       4 : [Ee]        5 : (Ee)[0-0]
# 2 : (exist)[.](free num)      3 : (none)[.](essential num)
#
# () : able to skipped, -- : essential, == : optional
# order one : (0) -- 1 -- 2 == 4 -- (0) -- 5
# order two : (0) -- 3 == 4 -- (0) -- 5
#

def floatOrNot(x):
    idxRe = 0
    for idx, token in enumerate(x):
        if token in ['+', '-']:         # sign check
            if idxRe == 0:
                continue
            elif idxRe == 4:
                idxRe = 5
                continue
            else:
                return (False, f"unproper location of {token}")

        if token in '0123456789':       # number check
            if idxRe == 0:
                idxRe = 1
                continue
            elif idxRe in [1, 2, 3, 5]:
                continue
            elif idxRe == 4:
                idxRe = 5
                continue

        if token == '.':                # dot check
            if idxRe == 0:
                idxRe = 3
                continue
            elif idxRe == 1:
                idxRe = 2
                continue
            else:
                return (False, "Alreay being dot")

        if token in ['e', 'E']:         # exponential notation check
            if idxRe in [2, 3]:
                idxRe = 4
                continue
            else:
                return (False, "No dot or No num")

        return (False, f"Not number {token}")               # no matching
    return (True, ) if idxRe > 1 else (False, "Int")        # float must have dot

#
# function floatOrNot return only True value as tuple type when input is float.
# but when input is not float then return False value and why False
#

X = list(map(str, X))
P = sum([[floatOrNot(x)] for x in X], [])
correct = [y == p[0] for y, p in zip(Y, P)]
accuracy = correct.count(True) / len(correct)

print(
    f"\nPerformance is {accuracy*100}%\n\nuncorrect result(case, label) is", end=''
)
if int(accuracy*100) == 100:
    print(" None.\n")
    print("[Result]")
    for i in range(len(P)):
        print("%10s"%str(X[i])+"%10s"%str(P[i][0])+"\t"+(str(P[i][1]) if P[i][0] != True else ''))
else:
    print()
    for i in range(len(correct)):
        if correct[i] != True:
            print(f"{[(X[i], Y[i], P[i]) ]}")

```

딱 100줄 코드\(주석이 좀 많긴 하지만 하하..\) 출력 결과도 깔끔하게 했다. 잘 결한 듯 싶다.



