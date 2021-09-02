---
description: '210827'
---

# DAY 5 :  Facenet \| Save

## Facenet

현재 데이터셋은 가로 384, 세로 512의 크기를 가졌는데, GPU 리소스 문제때문에 이미지를 충분히 줄여야 한다.

* 이미지 크기가 크면 한번에 올릴 수 있는 이미지가 적어져 batch\_size가 작아진다
* 마찬가지로 num\_workers 역시 줄어들게된다.

이 때 이미지의 크기를 줄이는 것은 이미지의 정보가 상실될 수 있으므로 어느정도 성능 하락의 요인이 될 수 있다. 그러면 이미지 사이즈를 줄이면서 정보는 최대한 유지하려면 어떻게 해야할까? 바로 우리가 구별하고자 하는 대상만을 최대한 담는것이다.

![&#xC778;&#xBB3C; &#xC0AC;&#xC9C4; &#xC608;&#xC2DC;](../../../.gitbook/assets/image%20%281020%29.png)

위와 같이, 인물 사진에 대해서 상하좌우로 배경또는 옷이 존재하고 이 정보는 우리가 이 인물이 여성인지 남성인지, 또는 나이가 30대 이상인지 이하인지 구분하는데 필요하지 않다.

* 패션 스타일로 구분할 수는 있겠지만 주관적인 요소라는 것을 떠나 모델이 그런 센스가 있을까?

따라서 이 얼굴을 중심으로 이미지를 Crop 하려고 한다. 물론 `torchvision.transform` 에서 `CenterCrop` 을 할 수도 있다. 다만 주어진 데이터셋이 다음 이미지와 같이 얼굴이 항상 이미지의 가운데만 존재하지는 않았다.

![](../../../.gitbook/assets/image%20%281016%29.png)

따라서, 얼굴을 기준으로 나누는 것이 합리적이다. 그리고 이 때 모든 이미지에 대해 얼굴을 찾지 못할 수도 있으므로 이에 대한 대책을 마련해야 한다.

얼굴을 왜 찾지못했을까? 모든 사진은 전면을 바라보고 있고 얼굴이 잘리지 않았다. 예상할 수 있는 이유는 마스크때문이다. 보통의 face detector는 dlib 라이브러리의 landmarks를 사용한다. 이에 대한 설명은 [\[dlib\] landmark 작동방식](https://sangmandu.gitbook.io/til/2021/mar/28) 에서 볼 수 있다. 다음 코드에서도 얼굴을 잡지 못한 이미지의 대부분이 마스크를 낀 이미지에서 발생함을 알 수 있다.

```python
import os
import cv2
from tqdm import tqdm
import torch
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from facenet_pytorch import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
TRAIN_IMAGE_DIR = "/opt/ml/input/data/train/images"
TRAIN_FACE_DIR = "/opt/ml/input/data/train/faces"
TEST_IMAGE_DIR = "/opt/ml/input/data/eval/images"
TEST_FACE_DIR = "/opt/ml/input/data/eval/faces"
```

```python
dirs = os.listdir(TRAIN_IMAGE_DIR)
dirs = [item for item in dirs if item[0] == "0"]
dir_bar = tqdm(dirs)
length = len(dirs)
x = y = 0
image_cnt = dict({'mask1' : 0,
                 'mask2' : 0,
                 'mask3' : 0,
                 'mask4' : 0,
                 'mask5' : 0,
                 'normal' : 0,
                 'incorrect_mask' : 0})
for idx, d in enumerate(dir_bar):
    path = os.path.join(TRAIN_IMAGE_DIR, d)
    
    for image_path in glob(os.path.join(path, '*')):
        file_name = image_path.split('/')[-1]
        file_name = file_name.split('.')[0]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        boxes, probs = mtcnn.detect(image)
        if boxes is None:
            dict[file_name] += 1
        else:
            x1, y1, x2, y2 = map(int, boxes[0])
            x1 = max(0, x1-30)
            y1 = max(0, y1-60)
            x2 = min(384, x2+30)
            y2 = min(512, y2+30)
            image = image[y1:y2, x1:x2]
            x += (x2-x1) 
            y += (y2-y1) 

    dir_bar.set_description(f'{idx} / {length}')
    
print(x / (7 * length), y / (7 * length))
for k, v in image_cnt.items():
    print(k, ":", v)
```

* 1-6 : 이미지에 접근하기 위한 경로를 준비한다.
* 20-21 : 이미지를 불러오고 RGB 배열로 변환한다.
* 23-25 : facenet 모델을 통해 탐지한 얼굴의 바운딩 박스를 얻는다. 만약 박스가 없으면 해당 종류를 카운트한다.
* 26-32 : 바운딩 박스좌표는 왼쪽 위 좌표와 오른쪽 아래 좌표로 이루어져있다. 이 두 좌표만 있으면 직사각형을 그릴 수 있기 때문이다. 모델이 잡은 얼굴이 조금 tight 할 수 있으므로 좌표를 조금 더 넓혀서 이미지를 Crop한다. 여기서는 양쪽 가로로 30만큼, 아래로 30, 위로 60만큼 늘렸다. 위로 더 늘린 이유는 헤어스타일도 남녀와 연령을 구분하는데 하나의 피처로 작용할 것이라 생각해서이다.
* 33-34 : 보통 탐지된 얼굴들의 크기의 평균을 구한다. 이 평균은 얼굴을 탐지하지 못한 이미지에 대해서 Crop 과 Resize에 사용할 것이다.
* 학습 데이터와 테스트 데이터에 대해서 이미지를 저장하는 코드는 생략했다.

![](../../../.gitbook/assets/image%20%281018%29.png)

평균적으로 가로는 213, 세로는 283의 크기를 가진다. \(이 크기는 가로로 +60, 세로로 +90이 반영된 수치이다. 평균적인 이미지는 이정도의 크기를 가진다\)

![](../../../.gitbook/assets/image%20%281015%29.png)

좀 더 안정성을 높이기 위해 얼굴을 탐지하지 못한 이미지에서 평균값의 1.2배만큼을 Crop하려고 한다. 반올림해서 가로는 256, 세로는 340의 크기를 가지려고 한다. 이미지의 중앙에서 Crop할 것이므로 이미지의 중앙 좌표\(세로,가로\) = \(256, 192\) 에서 더하고 뺀 영역을 Crop한다.

또한, 추후에 \(세로, 가로\) = \(280, 210\)으로 Resize 할 것이다.

![](../../../.gitbook/assets/image%20%281017%29.png)

이후, 이미지를 변환해 저장한다.

![](../../../.gitbook/assets/image%20%281019%29.png)



이후, 이전에 사용했던 csv 파일들의 path를 변경해준다.

```python
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

def face_path(x):
    return x.replace('images', 'faces')
train_df.path = train_df.path.apply(face_path)
test_df.path = test_df.path.apply(face_path)

train_df.to_csv('train_face.csv', index=False)
test_df.to_csv('test_face.csv', index=False)

train_df
```

![](../../../.gitbook/assets/image%20%281014%29.png)

경로가 images에서 faces로 바뀐 모습





## Save

지금까지는 매번 epoch를 하나 또는 두개씩 돌려보면서 성능이 좋은 모델을 찾았는데, 그 이유는 오버피팅 되는 시점이 매우 빨리왔기 때문이다. 적어도 6-7번이면 train\_data의 f1 score가 90점 후반대의 위치했다.

그러나, 학습률 스케쥴러를 사용하면서 적은 학습률에서도 학습을 하기 시작했고, 그에 따라 epoch수가 3배정도 늘어나게 되었기 때문에, 좋은 성능을 내는 모델의 파라미터를 저장하는 작업을 자동화 할 필요가 있었다.

여기서 좋은 성능이라 함은 valid\_data의 f1 score가 90점이 넘을 경우로 정의했다. 물론 가장 최고의 성능을 내는 모델만 저장할 수도 있지만, 실제 테스트 데이터에 대해서 좋은 성능\(=일반화\)을 낼지는 모르는 부분이므로 이렇게 판단했다. 또 90점이라는 커트라인을 정하면서 오버피팅 시키지 않으면서 조금이나마 일반화 능력을 챙기려는 의도도 들어갔다.

* 아니 valid\_data의 f1 score가 90점 이상인게 말이돼? 무슨 오버피팅? 이라고 할 수 있는데, 주어진 데이터셋이 모두 비슷한지, 생각보다 점수가 매우 잘 나온다. 팀원 중에는 train-valid 점수가 100-100 을 기록하기도 했다.
  * 어느정도냐면, 30에포크 정도 돌리면 10 에포크부터는 valid dataset의 f1 score가 97정도가 되서 20개 이상의 모델이 저장되버린다... 커트라인을 95점으로 올리고 싶은 욕구도 있지만, 테스트 셋에 대해서는 성능이 어떨지 모르니 좀 더 넓게 봤다.
* 그와 반면해, 테스트 데이터는 어려운 이미지가 존재하는 것이 분명했다. 100-100의 모델의 성능은 70점에 그쳤다.

```python
if valid_batch_f1/len(valid_loader) >= 0.9:
    torch.save(model.state_dict(), f'v:f1_{valid_batch_f1/len(valid_loader):.3f}_t:f1_{train_batch_f1/len(train_loader):.5f}_efficientnet_b4_state_dict.pt')  # 모델 객체의 state_dict 저장
```

계속 저장되는 파라미터들...

![](../../../.gitbook/assets/image%20%281026%29.png)

