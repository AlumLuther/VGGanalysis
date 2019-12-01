## 基于VGG16_BN的参数统计

#### 卷积层工作

1. 权重的数量级统计（PDF，CDF）
2. 卷积核L1范数的数量级统计（PDF，CDF）
3. 卷积核之间的皮尔逊相关系数（热图）

#### BN层工作

1. γ参数的数量级统计（PDF，CDF）

#### 存在的一些问题

- 量化不够细致，某些折线图形状略为极端；
- 热图横纵坐标过多（最大512×512），视觉效果并没有那么好；
- 以及对python和matplotlib的不熟练。