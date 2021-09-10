---
description: '210910'
---

# \[필수 과제 4\] Preprocessing for NMT Model

본 과제는 외부에 있는 코드가 아니므로 전체 분석을 할 수 없다. 다루고 싶은 일부분의 코드만 다루려고 한다.

### Typing

```python
from typing import List, Dict, Tuple, Sequence, Any
from collections import Counter, defaultdict, OrderedDict
from itertools import chain
import random
random.seed(1234)

import torch
```

* 처음보는 라이브러리 `typing` 이 있다. 아래 설명은 [여기](https://www.daleseo.com/python-typing/)를 참고했다.
* 타입 어노테이션을 사용하다보면 파이썬 내장 자료 구조에 대한 타입을 명시해야 할 때 사용하는 라이브러리이다.
  * 어노테이션은 함수의 인자가 어떤 타입인지를 명시해주는 가독성을 위한 코드이다. 어노테이션을 추가한다고 해서 언어 차원에서 어떤 실질적인 효력이 발생하는 것은 아니다.
* 예를 들어, 다음과 같이 사용가능하다.

```python
from typing import List
nums: List[int] = [1, 2, 3]

from typing import Dict
countries: Dict[str, str] = {"KR": "South Korea", "US": "United States", "CN": "China"}

from typing import Tuple
user: Tuple[int, str, bool] = (3, "Dale", True)
```



### Bucketing

Bucketing은 문장의 길이에 따라 데이터를 그룹화하여 padding을 적용하는 기법이다. 이 기법을 사용하기 전에는 어땠을까? 데이터셋에서 길이가 제일 큰 데이터에 모두 길이를 맞추어 &lt;pad&gt; 토큰을 추가했을 것인데, 만약 문장들이 평균적으로 길이가 짧고 가장 긴 문장이 outlier 처럼 유난히 긴 문장이라고 하면 어떨까? 대부분의 문장이 많은 &lt;pad&gt; 토큰을 추가하게 될 것이다.

이러한 문제를 해결하기 위해 Bucketing이 등장했다. 배치 사이즈 만큼 그룹핑해서 각 배치 사이즈에서 가장 긴 문자열에 맞추는 방식이다.

![https://livebook.manning.com/book/natural-language-processing-in-action/chapter-10/](../../../.gitbook/assets/image%20%281125%29.png)



### Collate Function

위 Bucketing을 구현해주려면, 각 데이터를 정렬하고 배치 사이즈를 고려해서 각 배치의 최대 길이로 맞춘 뒤 반환해줘야 한다. 이렇게 데이터를 원하는 형태의 batch로 가공하기 위해 사용하는 함수를 `collate` 라고 한다. 표준 라이브러리로 정의되어있기도 하고 커스텀해서 만들어 사용하기도 한다.













