## 深度学习基础

> 学习来源:
> * [计算机视觉与深度学习 北京邮电大学 鲁鹏](https://www.bilibili.com/video/BV1V54y1B7K3/?spm_id_from=333.337.search-card.all.click&vd_source=6eeaf968db275442f9be23b6183a3fd2)
> * [动手学深度学习 李沐](http://zh-v2.d2l.ai/)

### CV

#### 图像表示: vector
* Binary: 0/1
* Gray Scale: 0~255 1Byte
* Color: 3Byte RGB

#### 线性分类器

* 定义第 $i$ 类的线性分类器: $f_i(x, w_i^T) = w_i^Tx + b_i$

* 判断为第 $i$ 类规则： $if f_i(x) > f_j(x)$  $\forall i\neq j$
* 损失函数：度量给定分类器预测值与真实值不一致程度，通常输出为非负
    * $L = \frac{1}{N} \sum\limits_{i} L_i(f(x_i, w), y_i)$
    * 多类支撑向量机损失
        * 第 $i$ 个样本是第 $j$ 类的判定得分为： $S_{ij} = f_j(X_i, w_j, b_j) = W_j^TX_i+ b_i$
        * $$L_i = \sum\limits_{j \neq Y_i} 
\begin{cases} 0, if S_{yi}\geqslant S_{ij} + 1 \\
                S_{ij} - S_{yi} +1, otherwise
\end{cases} 
$$

        * 解释：如果真实类别的得分大于第i个样本预测得分则无损失；否则有损失
    * $L2$正则损失： $$L(w) = \frac{1}{N} \sum\limits_{i} L_i(f(x_i, w), y_i) + \lambda R(w)\\ 其中R(w) = \sum\limits_{k}\sum\limits_l|w_{kl}|^2$$
        * L2正则损失对大数值权值进行惩罚，喜欢分散权值，鼓励分类器将所有维度特征都用上。（单纯从效果上看，一定程度上避免了过拟合，使得分类器的分类能力更具泛化效果）
    * $L1$ 正则损失： $R(w) = \sum\limits_{k}\sum\limits_l\left |w_{kl}  \right |$

* 优化算法-梯度下降算法
    * 方向：负梯度方向，步长：learning rate
    * training_set: 训练参数
    * validate_set: 选超参
    * test_set: 评估模型泛化能力
    * K折交叉验证：验证集数据少

#### 全连接神经网络
$$
f = W_2max(0, W_1x + b_1) + b_2 ...
$$
* $W_1$ : $template$ , 人为指定； $W_2$ :融合多个模板匹配结果
* 为什么非线性操作  $（max）$ 不能去掉？
    *  $f = W_3(W_2(W_1x + b_1) + b_2) + b_3 = W^{'}x + b^{'} $ 
    , 与单层线性模型效果一致 
* 如何度量分类器输出与预测值之间的距离？
    * 熵： $H(p) = -\sum\limits_{x}p(x)logp(x)$
    * 交叉熵： $H(p, q)= -\sum\limits_{x}p(x)logq(x)$
    * 相对熵：用来度量两个分布之间的不相似性：
     $KL(p||q) = -\sum\limits_{x}p(x)log\frac{q(x)}{p(x)}$
     >  $$H(p, q)= -\sum\limits_{x}p(x)logq(x) = -\sum\limits_{x}p(x)logp(x) - \sum\limits_{x}p(x)log\frac{q(x)}{p(x)}\\ = H(p) + KL(p||q) \\ 
     而真实结果p，其熵为0，\\ 于是真实结果与预测结果之间的距离为其KL散度 $$
    *   $L_i = -log(\frac{e^{S_{yi}}}{\sum\limits_je^{S_j}})$
        * 个人理解，采用e的指数化是为了扩大数据之间的差距
        * 采用 $one-hot编码： L_i = -log(q_i)$
    * 与多类支撑向量机损失相比：交叉熵损失可充分利用预测正确的数据，使用   $Cross-entropy$  ,不但要求数据预测正确，而且要求正确类别的得分“高”
* 优化算法
    * 动量法
    * AdaGrad
    * RMSProp
    * Adam
* 初始化参数 Xavier：好的初始化方法可防止forward中信息小时，backward中梯度消失
* Dropout: 让隐层神经元以一定概率不被激活，应对过拟合
    * 使得神经元记录了更多信息，从效果上看起到了与正则化相似的作用
    * 可看做模型集成，仅在训练阶段使用

    
#### 卷积神经网络

* 采用平均卷积和，卷积后的图像产生了水平、竖直的条状
* 高斯卷积和：根据邻域像素与中心的远近程度分配权重  $G_{\sigma} = \frac{1}{2 \pi \sigma^2 } e^{- \frac{(x^2 + y^2)}{2\sigma^2}} $
    1. 确定卷积核尺寸
    2. 设置高斯函数标准差 $\sigma$
    3. 计算各位置的权重
    4. 对权重值归一化： 
    $\frac{G_\sigma(x, y)}{\sum\limits_{(x, y)}G_\sigma(x, y)}$
    > 权重归一化不会对信号衰减或增强

* 方差变化：方差越大，平滑效果越明显
* 尺寸变化：模板尺寸越大，平滑效果越强。
* 卷积核参数：大方差 or 大尺寸卷积核平滑能力强
* 高斯卷积核：去除图像中高频成分
* 可分离：可分解为两个以为高斯的乘积，小模板可减小计算量
* 边缘检测：图像中亮度明显而剧烈的变化的地方
    * 图像求导公式：  $\frac{\partial f(x, y)}{\partial x} \approx f(x + 1, y) - f(x, y) $
    *     
   $$
   \begin{bmatrix}
   1  \\
   -1 
   \end{bmatrix} 横向明显$$
   
   * $$
   \begin{bmatrix}
   1  &
   -1 
   \end{bmatrix}纵向明显$$

    * 卷积具有结合性： 
    $\frac{d}{dx}(f \ast g) = f\ast (\frac{d}{dx} g), 其中 \frac{d}{dx} g为高斯一阶偏导核$
    * 小方差高斯一阶偏导核模板可检测细粒度边缘信息
* Canny 边缘检测器
* 纹理表示：基于卷积核组的纹理表示方法
    * 思路：利用卷积核组提取图像中的纹理基，利用基元的统计信息表示图像的纹理
    * 对纹理分类任务：忽略基元位置，关注哪种基元对应的纹理以及对应基元出现的频率
        1. 设置卷积核组
        2. 卷积核组 -> 特征响应图
        3. 利用特征响应图的某种统计信息表示图像中的纹理

* 卷积操作： 
$W^Tx + b$ , $W$ 为卷积核权值， $b$ 为卷积核偏置
    * input:  $W_1 \times H_1$ output: $W_2 \times H_2$
    * 特征响应图的深度 = 卷积核个数 ：K
    * 卷积核尺寸：F
    * 卷积步长 Stride ： S
    * 边界填充 Padding ： P
    * 输入输出关系： 
    $$W_2 = (W_1 - F + 2P) / S + 1 \\ H_2 = (H_1 - F + 2P) / S + 1$$
    * Consider: 
        1. 卷积核宽、高
        2. padding
        3. stride
        4. 卷积核个数
* 池化操作：对每个特征响应图独立进行，降低特征响应图宽度和高度，减少后续卷积层参数的数量。
    * MaxPool 、AvgPool
    * 在该区域指定一个值代表该区域，计算量降低，感受野增大
* 图像增强 solve 过拟合


### NLP

#### RNN

#### Attention

#### Transformer
