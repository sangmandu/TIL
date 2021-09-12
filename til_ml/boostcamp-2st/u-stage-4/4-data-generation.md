---
description: '210824'
---

# \(4강\) Data Generation

### Data Feeding

Feed = 먹이를 주다 = 대상의 상태를 고려해서 적정한 양을 준다

![](../../../.gitbook/assets/image%20%28975%29.png)

모델의 성능과 데이터 제너레이터의 성능 중 어느 하나만 뛰어나서는 성능이 높아지지 않는다. 즉, 효율이 나오지 않는다. 따라서 서로의 성능을 고려해서 작동해야 한다.



### torch.utils.data

Dataset 구조

![](../../../.gitbook/assets/image%20%28973%29.png)

* init : dataset이 선언될 때 실행된다
* getitem : index로 아이템에 접근 가능해야한다
* len : 전체 dataset의 길이를 출력

DataLoader

![](../../../.gitbook/assets/image%20%28977%29.png)

* num\_workers : 데이터셋 처리를 병렬처리로 할 수 있게 한다. 이 때의 병렬 수
  * 많이 쓴다고 성능이 좋아지는 것은 아니므로 하나씩 올리면서 좋은 위치를 찾아야 한다
* drop\_last : 마지막 남은 배치를 살릴 지 말지 결정



Dataset은 바닐라 데이터를 원하는 형태로 출력해주는 클래스이며 DataLoader는 Dataset을 더 효율적으로 사용할 수 있게한다. 바닐라 데이터가 이미 원하는 형태의 데이터라면 Dataloader에 바로 바닐라 데이터를 입력해도 된다.



