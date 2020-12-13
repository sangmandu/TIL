---
description: TIL
---

# 13 Sun

## 오토마타와 컴파일러

### River Crossing : 최종 코드

{% embed url="http://lg-sl.net/product/scilab/sciencestorylist/ALMA/readSciencestoryList.mvc?sciencestoryListId=ALMA2020010003" %}

![](../../.gitbook/assets/image%20%2847%29.png)

```text
#
#   data : 2020.12.02
#   author : sangmandu at Dankook Univ.
#   program : river crossing
#

#
#   H : Human,  W : Wolf,   S : Sheep,  C : Cabbage
#   1 ... W eat S when there are only W and S
#   2 ... S eat C when there are only S and C
#   3 ... boat can be accommodate in up to 2 elements and Human essential
#   4 ... objective is that all element moves left to right
#

#
#   two lists that left and right
#   prior : keep from generating duplicate
#   dir : direction left to right or vice versa
#   record : the procedure to success(or failure)
#
#   failure     1) left check issue : collision with rule 1 or 2
#               2) left check issue : collision with rule 1 or 2
#               3) duplicate case : prevent that infinite loop
#   

# initialization
_left = ['H', 'W', 'S', 'C']
_prior = ''
_right = []
_dir = '>'
_record = []

# check rule 1 and 2
# when breaking rule then True else False
def checkIssue(rand):
    if len(rand) == 2 and 'S' in rand:
        if 'W' in rand or 'C' in rand:
            return True
    return False

stack = [(_left, _prior, _right, _dir, _record)]
while stack:
    left, prior, right, dir, record = stack.pop(0)
    print()
    print("selected element : ", left, prior, right, dir)
    print("and remained stack : ", stack)
    if not left:
        _record.append(record)
        continue

    if dir == ">":
        for ele in left:
            if ele == 'H':
                continue
            print(f"ele is {ele}")
            l, p, r, rec = left[:], prior[:], right[:], record[:]
            if ele == prior:
                print("ele == prior")
                continue
            l.remove(ele)
            l.remove('H')
            if checkIssue(l):
                print("left check issue")
                continue
            r.append(ele)
            r.append('H')
            if checkIssue(r):
                print("right check issue")
                continue
            memo = f"{ele} and human move from left : {left} to right : {right}"
            if memo in rec:
                print("duplicate issue")
                continue
            rec.append(memo)
            stack.append((l, ele, r, '<', rec))
            print(f"stack append {(l, ele, r, '<', rec)}")
    else:   # dir == "<"
        for ele in right:
            if ele == 'H':
                continue
            print(f"ele is {ele}")
            l, p, r, rec = left[:], prior[:], right[:], record[:]
            if ele == prior:
                print("ele == prior")
                continue
            r.remove(ele)
            r.remove('H')
            if checkIssue(r):
                print("right check issue")
                continue
            l.append(ele)
            l.append('H')
            if checkIssue(l):
                print("left check issue")
                continue
            memo = f"{ele} and human move to left : {left} from right : {right} "
            if memo in rec:
                print("duplicate issue")
                continue
            rec.append(memo)
            stack.append((l, ele, r, '>', rec))
            print(f"stack append {(l, ele, r, '>', rec)}")
        record.append(f"only human moves to left : {left} from right : {right} ")
        right.remove('H')
        left.append('H')
        if checkIssue(right):
            print("right check issue when human only on board")
            continue
        stack.append((left, '', right, '>', record))
        print(f"stack append that only human {(l, '', r, '>', rec)}")

print()
for idx, record in enumerate(_record, 1):
    print(f"#{idx} success case")
    for rec in record:
        print(rec)
    print()
```

중간 코드와는 비슷하면서 좀 다르다. \(한번 갈아 엎어가지고...\) 재귀 함수보다는 while / stack 방식이 변수 관리가 편한 것 같다. 함수 내에서 반복적으로 사용되는 변수는 l = left\[:\] 와 같이 깊은 복사를 하려고 해도 잘 안된다. 결국 copy.deepcopy를 써야 되는데, 굳이 이렇게? 라는 생각이 너무 많이 든다. 함수 내에서는 재귀함수 또한 동일한 변수를 사용해서 그런게 아닐까 라는 추측은 있지만 확실한 이유는 모르겠다.

100줄 조금 넘게 작성된 코드. 좀 더 함수화하고 함축하여 코드를 줄일 수 있겠지만 애초에 메모리가 작아서 그럴 필요가 없다. 좀 더 직관적이어서 설명하거나 볼 때는 좋을 수 있다. 조금 더 수정하자면 prior 변수를 없애고 duplicate case에 추가할 수 있겠다.

결과도 150줄 정도 출력된다. 각 case를 BFS 방식으로 접근하며 각 케이스마다 실패할 경우 실패 이유가 출력된다. 실패했을 경우에는 그 과정은 일부로 출력하지 않도록 했다. \(안궁금해서?\) 또한 성공할 경우를 마지막에 정리해서 다시 출력하도록 했다. 실제로도 2가지의 경우의 수가 존재하는데, 프로그램 또한 두 가지의 성공 케이스를 출력한다.

![](../../.gitbook/assets/image%20%2849%29.png)

BFS방식으로 출력하는 모습. 각 실패 케이스의 이유는 다음과 같다.

![](../../.gitbook/assets/image%20%2851%29.png)

다음은 성공적으로 출력되는 모습.

![](../../.gitbook/assets/image%20%2852%29.png)

