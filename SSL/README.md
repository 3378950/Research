## Semi-Supervision-Learning

### Overview
> Reference: 《机器学习-周志华》
1. 使用未标记的样本
    * 有训练样本集 $D_l = \{ (x_1, y_1), (x_2, y_2), ..., (x_l, y_l) \}$ , 均为有标记样本
    * 另外还有 $D_u = \{ (x_{l + 1}, y_{l + 1}), (x_{l + 2}, y_{l + 2}), ..., (x_{l + u}, y_{l + u}) \}, l \ll u$ ,这u个样本均为类别标记未知
    * 若直接使用传统监督学习技术，则仅有 $D_u$ 能用于构建模型 $D_u$ 所包含的信息被浪费了.另一方面，若 $D_u$ 较小，则由于训练、样本不足，学得模型的泛化能力往往不佳
    * 分类与分布：有标记的样本已经有了明确的分类，而对于无标记的样本，若他们与有标记样本是从同样的数据源独立同分布采样而来，它们所包含的数据分布的信息对模型是有意义的
    * 未标记样本所揭示的数据分布信息与类别标记相联系的假设：**相似的样本拥有相似的输出**
    * 半监督学习可划分为纯半监督学习和直推学习
    ![SSL-classification](../imgs/SSL.PNG)

2. 生成式方法
...


### SSL-models
***
* [UDA](https://arxiv.org/pdf/1904.12848v6.pdf): Unsupervised Data Augmentation for Consistency Training
    ![](https://img-blog.csdnimg.cn/20210321123929596.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3plcGh5cl93YW5n,size_16,color_FFFFFF,t_70)
    * 关键：对无标记数据进行数据增强
    * 效果：使用很少的标注数据，就可达到与监督学习媲美甚至超越的效果
    * 数据增强
        * RandAugment：Python Image Library(PIL)中均匀采样数据增强方法
        * Back Translation：语言A翻译到语言B，再从B翻译回语言A
        * TF-IDF分的单词替换掉，保持高TF-IDF分
    * 一致性损失函数
    ![](https://img-blog.csdnimg.cn/20210321123857131.png)
    
    * Trick in training ...
***
