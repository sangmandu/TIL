---
description: '210820'
---

# \(10강\) PyTorch Troubleshooting

### OOM, Out Of Memory

왜 발생했는지, 어디서 발생했는지 알기가 어렵다. Error backtracking이 프로그램의 기저까지 들어가기 때문에 이상한데로 감

배치사이즈를 줄이고, GPU를 초기화한다음에 다시 작동시키는것이 기본적인 해결방법



그 외의 방법

* GPUUtil
  * nvidia-smi 처럼 GPU 상태를 보여주는 모듈
  * iter마다 메모리가 늘어나는지를 확인할 수 있다

```python
!pip install GPUtil

import GPUtil
GPUtil.showUtilization()
```

* torch.cuda.empty\_cache\(\)
  * 사용되지 않는 GPU상 cache를 정리한다
  * 가용 메모리를 확보할 수 있다
  * del과 같은 기능은 아니다
  * reset 대신 쓰기 좋은 함수이다
* training loop에 tensor로 축적되는 변수를 확인할 것
* del 명령어를 적절히 사용하기
* 가능한 batch 사이즈 실험해보기
* torch.no\_grad\(\) 사용하기





