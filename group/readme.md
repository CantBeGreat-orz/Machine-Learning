
### 概况


- 编写实现了指定模型（for task 5），成功在给定样本上进行了训练和测试，且不只适用于给定样本

- 额外完成了**高斯混合模型的EM算法**与对**Libsvm**的学习使用

- 其中前向神经网络支持**自定义隐层节点数**，使用的激活函数为sigmoid函数

- 分类算法支持自定义学习率衰减与批大小

- 聚类算法支持**三种初始节点生成方式**：

  （1）从样本点中随机选取（并计算参数  for高斯混合模型的EM算法）

  （2）在样本空间（参数空间）中随机生成

  （3）kmeans++方法（在keams模型中）；内置kmeans预处理获得中心点计算得到参数（高斯混合模型的EM算法）


#### 项目结构

├── group

  ├── model

  │  ├── fnn.py # 前向神经网络模型

  │  ├── gmms.py    # 高斯混合模型的应用（贝叶斯与高斯混合模型的EM算法）

  │  ├── kmeans.py # K均值聚类

  │  ├── pla.py # 多分类感知机

  │  ├── softmax.py # Softmax回归

  ├── utils

  │  ├── function 

  │  │  ├── gmm.py  # 高斯混合模型

  │  │  ├── predict.py # 预测函数

  │  │  ├── sigmoid.py  # sigmoid函数

  │  ├── preprocess

  │  │  ├── addbias.py # 增加偏置项

  │  │  ├── normalize.py  #归一化

  ├── colormap.py        # 可视化颜色map

  └── main.py       # 主函数

------

### 主函数（*main.py*）简介

#### 参数说明

| 参数名       | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| *alpha*      | 学习率（for 分类算法）                                       |
| *epochs*     | 时期数（for 分类算法）                                       |
| *steps*      | 迭代次数（每个时期中）（for 分类算法）                       |
| *mini_batch* | 批大小（for 分类算法）                                       |
| *decay*      | 衰减率（for 分类算法）                                       |
| *limit*      | 最小衰减到原学习率的百分比（for 分类算法）                   |
| *rand*       | 用来指示是否在样本空间随机生成初值（for 聚类算法）           |
| *plus*       | kmeans：指示是否使用kmeans++方法，优先级大于rand；gmm_em：指示是否使用kmeans初始化参数，优先级小于rand |
| *animation*  | 是否显示动画（for 聚类算法）                                 |
| *iter_times* | 迭代次数（for 聚类算法）                                     |

#### 获取模型

| 函数名              | 说明                     |
| ------------------- | ------------------------ |
| *get_model_pla*     | 获取感知机模型           |
| *get_model_softmax* | 获取Softmax回归模型      |
| *get_gmm_bayes*     | 获取贝叶斯模型           |
| *get_model_fnn*     | 获取前馈神经网络模型     |
| *get_kmeans*        | 获取kmeans模型           |
| *get_gmm_em*        | 获取高斯混合模型的EM算法 |

#### 运行模型

| 函数名               | 说明         |
| -------------------- | ------------ |
| *run_classification* | 运行分类模型 |
| *run_cluster*        | 运行聚类模型 |
| *run_svm*            | 运行svm      |

#### 使用方法

对要使用的模型，取消其获取模型的函数与对应的运行模型的函数的注释，调整参数，开始运行

------

### 编写环境

- Windows10
- vscode
- python 3.9.6

