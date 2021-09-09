---
description: '210910'
---

# \[필수 과제 4\] Preprocessing for NMT Model

본 과제는 외부에 있는 코드가 아니므로 전체 분석을 할 수 없다. 다루고 싶은 일부분의 코드만 다루려고 한다.



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

















