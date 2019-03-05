---
layout: post
category: "ml"
title: "特征工程-特征选择"
tags: [特征工程，特征选择，pearson相关系数, Pearson Correlation Coefficient]
urlcolor: blue
---

目录

<!-- TOC -->

- [1. 特征的统计指标](#1-特征的统计指标)
	- [1.1 连续型特征与连续型特征-相关性检验办法](#11-连续型特征与连续型特征-相关性检验办法)
	- [1.2 离线型特征与离线型特征-相关性检验办法](#12-离线型特征与离线型特征-相关性检验办法)
	- [1.3 连续型特征与离散型特征-相关性检验办法](#13-连续型特征与离散型特征-相关性检验办法)
	- [1.4 ](#)
	- [1.5 ](#)
	- [1.6 ](#)
- [2. 特征选择的方法](#2-特征选择的方法)
	- [2.1 过滤法filter](#21-过滤法filter)
	- [2.2 包装法wrapper](#22-包装法wrapper)
	- [2.3 嵌入法embedding](#23-嵌入法embedding)

<!-- /TOC -->

特征选择的目的是为了选择出能够有效区分目标(因变量)的特征(自变量)，那么什么是好的特征呢？

+ 特征越发散越是好特征：如果特征不发散，即标注差为0，说明该特征基本上没差异，不利于区分目标
+ 特征与目标越相关越是好特征

## 1. 特征的统计指标

### 1.1 连续型特征与连续型特征-相关性检验办法

#### 1.1.1 pearson/皮尔森相关系数:协方差与标准差的商
[协方差参考](https://www.zhihu.com/question/20852004)

它有多种表达形式，常见有如下两种：

> 公式1

$$\rho_{x,y}=\frac{cov(x, y)}{\sigma_{x}\sigma_{y}}=\frac{E((x-\bar{x})(y-\bar{y}))}{\sigma_{x}\sigma_{y}}=\frac{E(xy)-E(x)E(y)}{\sqrt{E(x^{2})-E^{2}(x)}\sqrt{E(y^{2})-E^{2}(y)}}$$

> 公式2

$$\rho_{x,y}=\frac{\sum xy-\frac{\sum x\sum y}{N}}{\sqrt{(\sum x^{2}-\frac{(\sum x)^{2}}{N})(\sum y^{2}-\frac{(\sum y)^{2}}{N})}}$$

> [其他公式参考](http://blog.csdn.net/zhangjunjie789/article/details/51737366)

```python
from math import sqrt

def multiply(a,b):
    #a,b两个列表的数据一一对应相乘之后求和
    sum_ab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sum_ab+=temp
    return sum_ab

def cal_pearson(x,y):
    n=len(x)
    #求x_list、y_list元素之和
    sum_x=sum(x)
    sum_y=sum(y)
    #求x_list、y_list元素乘积之和
    sum_xy=multiply(x,y)
    #求x_list、y_list的平方和
    sum_x2 = sum([pow(i,2) for i in x])
    sum_y2 = sum([pow(j,2) for j in y])
    molecular=sum_xy-(float(sum_x)*float(sum_y)/n)
    #计算Pearson相关系数，molecular为分子，denominator为分母
    denominator=sqrt((sum_x2-float(sum_x**2)/n)*(sum_y2-float(sum_y**2)/n))
    return molecular/denominator
```

> 当两个变量的标准差都不为零时，相关系数才有意义，皮尔逊相关系数适用于：
> 
> 1. 两个变量之间是线性关系，都是连续数据。
> 2. 两个变量的总体是正态分布，或接近正态的单峰分布。
> 3. 两个变量的观测值是成对的，每对观测值之间相互独立。

#### 1.1.2 spearman/斯皮尔曼相关系数

$${x}'=\frac{x-min(x)}{max(x)-min(x)}$$

### 1.2 离线型特征与离线型特征-相关性检验办法

### 1.3 连续型特征与离散型特征-相关性检验办法

### 1.4

### 1.5

### 1.6

## 2. 特征选择的方法

### 2.1 过滤法filter

### 2.2 包装法wrapper

### 2.3 嵌入法embedding


