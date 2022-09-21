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

* 定义第$i$类的线性分类器:$f_i(x, w_i^T) = w_i^Tx + b_i
$

* 判断为第$i$类规则：$if f_i(x) > f_j(x)$  $\forall i\neq j$
* 损失函数：度量给定分类器预测值与真实值不一致程度，通常输出为非负
    * $L = \frac{1}{N} \sum\limits_{i} L_i(f(x_i, w), y_i)$
    * 多类支撑向量机损失
        * 第$i$个样本是第$j$类的判定得分为：$S_{ij} = f_j(X_i, w_j, b_j) = W_j^TX_i+ b_i$
        * $$L_i = \sum\limits_{j \neq Y_i}
            \left\{
                \begin{aligned}
                0, if S_{yi}\geqslant S_{ij} + 1 \\
                S_{ij} - S_{yi} +1, otherwise
                \end{aligned}
            \right.  = \sum\limits_{j \neq y_i}max(0, S_{ij} - S_{yi} + 1)$$
        * 解释：如果真实类别的得分大于第i个样本预测得分则无损失；否则有损失
    * $L2$正则损失：$$L(w) = \frac{1}{N} \sum\limits_{i} L_i(f(x_i, w), y_i) + \lambda R(w)\\ 其中R(w) = \sum\limits_{k}\sum\limits_lw_{kl}^2$$
        * L2正则损失对大数值权值进行惩罚，喜欢分散权值，鼓励分类器将所有维度特征都用上。（单纯从效果上看，一定程度上避免了过拟合，使得分类器的分类能力更具泛化效果）
    * $L1$正则损失：$R(w) = \sum\limits_{k}\sum\limits_l\left |w_{kl}  \right |$

* 优化算法-梯度下降算法
    * 方向：负梯度方向，步长：learning rate
    * training_set: 训练参数
    * validate_set: 选超参
    * test_set: 评估模型泛化能力
    * K折交叉验证：验证集数据少

#### 全连接神经网络

#### 卷积神经网络