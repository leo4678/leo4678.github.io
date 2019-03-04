---
layout: post
category: "ml"
title: "FM（因子分解机）"
tags: [ml，FM]
urlcolor: blue
---

目录

<!-- TOC -->

- [1. 原理](#1-原理)
	- [1.1 定义](#11-定义)
	- [1.2 降低计算复杂度](#12-降低计算复杂度)
- [2. 应用：回归和分类](#2-应用：回归和分类)
	- [2.1 回归](#21-回归)
	- [2.2 分类](#22-分类)
- [3. 优化器](#3-优化器)

<!-- /TOC -->

非常好的两篇介绍FM算法原理及应用的文章

[分解机(Factorization Machines)推荐算法原理](https://www.cnblogs.com/pinard/p/6370127.html)

[知乎：推荐系统召回四模型之：全能的FM模型](https://zhuanlan.zhihu.com/p/58160982)

## 1. 原理

### 1.1 定义

现实世界中的大部分问题都是非线性关系（自变量和因变量不能表达成y=kx+b的形式），用机器学习的方法来处理预测任务时，需要使用非线性的模型来建模，非线性模型一般包括(树模型、深度学习模型等)。在工业界中，因线性模型的高效、简单、易扩展、容易解释等特点，从而获得了更广泛的应用。考虑实际问题的特性，工业界需要一种能够拟合非线性特点的线性模型。一种简答易行的方式是使用特征工程中的特征交叉，如：工业界应用的LR等，但是这种手工做特征交叉的方式会耗费大量的人力，那么**能不能找到一种从模型层面表达特征交叉的线性模型呢？答案是有的，这就是FM(Factorization Machines/因子分解机)**。

其表达形式如下：

$$y=w_{0}+\sum_{i}^{n}w_{i}x_{i}+\sum_{i}^{n}\sum_{j=i+1}^{n} <v_{i},v_{j}> x_{i}x_{j}$$

其中\\(v_{i}\\)为表达索引为i特征的一组隐向量，这种表达形式能够解决样本高稀疏数据带来的泛化能力差的缺点。

### 1.2 降低计算复杂度

原式中表达特征交叉\\(\sum_{i}^{n}\sum_{j=i+1}^{n} <v_{i},v_{j}> x_{i}x_{j}\\)的部分时间复杂度为\\(O(kn^{2})\\)，通过如下转换可以变成\\(O(kn)\\)

<html>
<br/>

<img src='/assets/FM算法降低复杂度推导.png' style='max-height: 754px;max-width:614px'/>
<br/>

</html>

## 2. 应用：回归和分类

### 2.1 回归

定义损失函数为最小平方误差（least square error），即：

$$loss(\hat{y},y)=(\hat{y}-y)^{2}$$

### 2.2 分类

对于二分类，损失函数可取hinge loss函数，logit loss 函数，cross entropy 函数。下式中\\(\hat{y}\\)为预测值，\\(y\\)为标签

#### hinge loss 函数

标签要求 y = -1 或者 +1

<html>
<br/>

<img src='/assets/hinge_loss.png' style='max-height: 400px;max-width:500px'/>
<br/>

</html>

#### logit loss 函数

标签要求 y = -1 或者 +1

$$loss(\hat{y},y)=-ln \sigma (\hat{y}y)$$

其中\\(\sigma (x)=\frac{1}{1+e^{-x}}\\)

#### cross entropy 函数

标签要求 y = 0 或者 1

$$loss(\hat{y})=-\hat{y}log(\hat{y})-(1-\hat{y})log(1-\hat{y})$$

## 3. 优化器

[参考该文第4部分](https://www.cnblogs.com/pinard/p/6370127.html)


