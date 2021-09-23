---
description: '210901'
---

# DAY 10 : Cutmix



![](https://lh5.googleusercontent.com/lR11yTImUL-TJNLn6p13AtnuEy5nvdcuh6lr0wUw6XTWirHgvRd-VVFLV-5pKWWnli_tw9ZFQZ1SrObXzTumATSPjNKULRVvKFuuy5DuyHDascRfNTi0xx726CpPDGUQ1OURLm8L=s0)

컷믹스를 하면 성능이 증가한다. 모든 로스함수에 대해서 실험해보았더니 모두 성능이 오르는 놀라운 결과를 보였다.

다음은 cutmix를 구현하는 코드이다.

```python
if np.random.random() <= args.cutmix:
    W = inputs.shape[2]
    mix_ratio = np.random.beta(1, 1)
    cut_W = np.int(W * mix_ratio)
    bbx1 = np.random.randint(W - cut_W)
    bbx2 = bbx1 + cut_W
    
    rand_index = torch.randperm(len(inputs))
    target_a = labels
    target_b = labels[rand_index]
    
    inputs[:, :, :, bbx1:bbx2] = inputs[rand_index, :, :, bbx1:bbx2]
    outs = model(inputs)
    loss = criterion(outs, target_a) * mix_ratio + criterion(outs, target_b) * (1. - mix_ratio)
```

* 1 : 인자로 `args.cutmix` 를 받는다. 0에서 1의 값을 가지며 이 값이 클수록 cutmix를 적용할 가능성이 증가한다.
* 3 : 베타분포에서 특정 수를 뽑는다, 두 인자의 수가 같으면 균등분포로 표현할 수 있다.

![](../../../.gitbook/assets/image%20%281205%29.png)

* 4 : cutmix를 적용할 비율을 결정했다면 이를 가지고 가로에 대해 자를 부분을 결정한다.
  * 실제로 cutmix는 임의의 w와 h의 이미지를 합치는 것
  * 여기서 데이터셋의 이미지를 보면 대체로 얼굴이 중앙에 있고 그 외에는 벽과, 옷이 있기 때문에 우리는 얼굴 데이터에 초점을 맞춰야한다. 또한, 얼굴이 있어야 할 위치에 벽이나 옷이 합성되면 안되기 때문에 세로길이는 이미지 세로 길이로 고정하고 가로 길이만 바꾸기로 한다.
* 5-6 : cutmix할 이미지의 시작점과 끝점
* 이후, cutmix를 적용한다.

