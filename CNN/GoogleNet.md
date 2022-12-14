## GoogleNet

### 解决的问题
* 提高了网络内部计算资源的利用率。在增加了网络深度和广度的同时保持了计算预算不变。


### 主要贡献
* 以Inception module的形式引入了一种新层次的组织方式，在更直接的意义上增加了网络的深度。
* 提出了一种Inception的深度卷积神经网络结构，并在分类和检测上取得了新的最好结果。


### 动机
* 通过增加网络的深度、宽度来提高网络的性能，但存在两个缺点：
    * 更多的参数，这会使增大的网络更容易过拟合
    * 计算资源使用的显著增加
* 解决方法：引入稀疏性并将全连接层替换为稀疏的全连接层，甚至是卷积层。
    * 存在的问题：目前的极端架构效率仍非常低下；非均匀的稀疏模型也要求更多的复杂工程和计算基础结构。
    * 目前大多数面向视觉的机器学习系统通过采用卷积的优点来利用空域的稀疏性。
* 作者希望：一个架构能利用滤波器水平的稀疏性，但能通过利用密集矩阵计算来利用我们目前的硬件。
