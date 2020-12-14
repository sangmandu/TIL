---
description: TIL
---

# 12 Sat

## 오토마타와 컴파일러

### River Crossing : 중간 코드

{% embed url="http://lg-sl.net/product/scilab/sciencestorylist/ALMA/readSciencestoryList.mvc?sciencestoryListId=ALMA2020010003" %}

![](../../.gitbook/assets/image%20%2848%29.png)

River Crossing 중간코드 리뷰

재귀가 너무 많아서 아직 실패하는 코드.

모든 케이스에 대해 성공 실패를 알려주며 각 결과에 대해 경로를 출력해주려고 한다. 오토마타에서 토큰 분석 및 제어 하는 부분을 확대한 문제. 

```text
left = ['W', 'S', 'C']
right = []


def pos(_left, _right):
    if 'S' in _left:
        if 'W' in _left or 'C' in _left:
            return False
    elif 'S' in _right:
        if 'W' in _right or 'C' in _right:
            return False
    return True

def cross(_left, _right, dir, record, element = None):
    if len(_right) == 3: return record
    if dir:    # src -> dst
        for ele in _left:
            if ele == element:
                continue
            l, r, rec = _left, _right, record
            l.remove(ele)
            if pos(l, r):
                r.append(ele)
                if pos(l, r):
                    rec.append(f"{ele} moves from src to dst")
                    record.append(cross(l, r, 0, rec, ele))
                else:
                    print("Failed")
                    print(record)
                    print()
            else:
                print("Failed")
                print(record)
                print()
    else:      # dst -> src
        for ele in _right:
            if ele == element:
                continue
            l, r, rec = _left, _right, record
            r.remove(ele)
            if pos(l, r):
                l.append(ele)
                if pos(l, r):
                    rec.append(f"{ele} moves from dst to src")
                    record.append(cross(l, r, 1, rec, ele))
                else:
                    print("Failed")
                    print(record)
                    print()
            else:
                print("Failed")
                print(record)
                print()
        record.append(cross(_left, _right, 0, record))

    return record

success = cross(left, right, 1, [])
print(success)
```

