## 机器学习
> Reference:  
> * [机器学习-白板推导](https://space.bilibili.com/97068901/channel/seriesdetail?sid=594044)
> * 《机器学习-周志华》


### [数学基础 - 高斯分布](./1.Math.md)
* 频率派与贝叶斯派
* 一维下的高斯分布 - 极大似然估计
* 多维情况的高斯分布
    1. 定性分析 - 概率密度角度
    2. 存在的问题 - 局限性
    3. 常用定理介绍
        - 已知联合概率求边缘概率及条件概率
        - 已知边缘和条件概率求联合概率分布
* 杰森不等式

### [线性回归](./2.Linear%20Regression.md)
* 最小二乘法
    *  $L2$ 范数定义的损失误差
    * 几何解释：求 $Y$ 向量到 $P$ 维空间的距离最小的向量
    * 噪声为高斯分布的 MLE：$LSE \rightleftharpoons MLE$
* 正则化
    * 正则化框架
    * 权重先验为高斯分布的MAP: $Regularized \space LSE \rightleftharpoons MAP$

