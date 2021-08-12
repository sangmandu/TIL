---
description: '210812'
---

# \(06강\) Computer Vision Applications

이전의 모델들은 이미지 분류 뿐만 아니라 Segmentaion과 Detection들에 대해서도 연구했다.

## Semantic Segmentation

이미지의 모든 픽셀이 어떤 라벨에 속하는지 찾는 것이다. Dense Classification

* 자율주행 문제등의 많이 사용된다.

### Fully Convolutional Network

전통적인 CNN은 이렇다.

![](../../../../.gitbook/assets/image%20%28840%29.png)

Fully convolutional network는 dense layer가 없다.

![](../../../../.gitbook/assets/image%20%28862%29.png)

dense layer를 없애는 과정을 convolutionalization 이라고 한다. dense layer가 없어진 자체가 장점.

![](../../../../.gitbook/assets/image%20%28842%29.png)

파라미터 수는 동일하다.

* 왼쪽 : 4x4\(필터\) x 16\(입력채널\) x 10\(출력채널\)
* 오른쪽 4x4\(필터\) x 16\(입력채널\) x 10\(출력채널\)

파라미터도 달라진 것이 없는데 왜 이렇게 할까?

기존 CNN은 결과를 1차원으로 뱉기 때문에 단순히 분류만 할 수 있었는데, FCN은 결과를 이미지로 뱉기 때문에 입력 이미지에 대한 히트맵을 얻을 수 있기 때문\(나도 확실하게 잘 모르겠다\)

또한, FCN은 어떤 크기의 입력이라도 상관없이 입력할 수 있지만 출력의 크기가 작아지게된다. 그래서 이러한 출력의 크기를 늘릴 수 있는 기술들이 필요했고 이러한 기술들이 등장하게 된다

### Deconvolution\(conv transpose\)

![](../../../../.gitbook/assets/image%20%28847%29.png)

convolution의 역연산을 해준다. 그러면 30x30이 15x15가 된 것을 다시 30x30으로 하게 해준다. 근데 사실 역연산은 불가능하다.

* 3 + 7 = 10 이고 2 + 8 = 10 이지만, 10을 가지고 원래 수가 무엇이였는지는 알 수 없기 때문

따라서 엄밀히 말하면 역연산은 아니다. 그렇지만 파라미터의 숫자와 네트워크의 구조를 봤을 때는 역연산이라고 볼 수 있다. 아래와 같이 패딩을 많이줘서 원래 크기로 돌리는 모습

![](../../../../.gitbook/assets/image%20%28855%29.png)

![](../../../../.gitbook/assets/image%20%28843%29.png)

## Detection

### R-CNN

할 수 있는 가장 간단한 방법이다. 이미지 내에서 2천개정도의 REGION\(Boundix box\)을 뽑아낸다. 각각의 뽑은 Region의 크기를 통일하고 CNN을 통해 특징을 얻은 다음 SVM을 사용해서 분류한다.

### SPPNet

Spatial Pyramid Pooling

R-CNN의 가장 큰 문제중 하나는 이미지 안에서 바우닝 박스를 2천개를 뽑으면 이 2천개를 다 CNN에 넣어야 되는 것. 이미지 한장을 위해 모델을 2천번 돌려야했다.

SPPNet의 아이디어는 CNN을 한번만 돌리자. 그리고 이미지 내에서 얻은 바운딩 박스에 해당하는 피처맵을 얻자.

* R-CNN 에대해서 빨라졌다.

### Fast R-CNN

SPP와 동일한 컨셉을 가졌다. 뒷단의 Neural Network를 통해서 Boudning box를 결정한다는 차이

### Faster R-CNN

Bounding box를 뽑아내는 Region Proposal도 학습을 하자는 아이디어. 왜냐면 이 Reigon을 뽑는 알고리즘은 임의의 Region을 뽑기 때문.

Region Proposal Network는 뽑은 Region안에 물체가 있을지 없을지를 판단한다. 여기서 필요한 것이 Anchor box이다

* Anchor box는 미리 정해놓은 bounding box의 크기이다

여기서도 FCN이 활용된다.

![](../../../../.gitbook/assets/image%20%28839%29.png)

넓이가 각각 128, 256, 512인 Bounding Box의 가로 세로 비율이 1:1, 1:2, 2:1이므로 총 9개의 Box가 존재한다.

또, Bounding box의 중심점 \(x, y\)와 가로와 세로길이 w, h의 4가지 파라미터가 필요하다.

Bounding box가 쓸모가 있는지 없는지에 대한 Yes or No의 2가지 파라미터가 필요하다

### YOLO

지금은 v5까지 나왔는데, v1을 다룰 것임

기존 분류기들과 다른점은 이미지의 Region을 찾고 이 Region에 해당하는 피처맵을 뽑는 것이 아니라 그냥 이미지 한장을 입력하면 바로 결과가 나올 수 있도록 하는 구조를 가졌다

이전에는 Region Proposal Network가 있었고 거기서 나오는 Bounding Box를 따로 분류했다. YOLO는 한번에 분류한다.

![](../../../../.gitbook/assets/image%20%28856%29.png)

YOLO는 이미지가 들어오면 SxS의 격자로 이미지를 나누게 된다. 찾고자 하는 물체의 중앙이 그리드 안에 들어가면 그 그리드셀이 해당 물체의 Bounding box와 Class까지 같이 예측해주게된다.

이전에는 Anchor box가 있었는데 여기서는 없고 단지 Bounding box의 개수만을 미리 정해주게 된다. \(논문에서는 5개\) 그러면 모델은 n개의 bounding box의 \(x, y, w, h\)를 찾게된다. 그리고 실제로 쓸모있는지에 대한 여부도 반환한다.

 결국 S\*S\*\(B\*N+C\) 의 크기를 갖는 결과가 반환된다.

* S\*S : 그리드의 개수
* B\*N : 바운딩 박스\(x, y, w, h, confidence\)와 미리 정의한 개수
* C : Number of Classes

















