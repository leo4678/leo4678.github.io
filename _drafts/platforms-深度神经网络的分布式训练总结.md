---
layout: post
category: "platforms"
title: "深度神经网络分布式训练方法"
tags: [深度网络分布式训练，分布式随机梯度下降的同步和异步变体，各种All Reduce梯度聚合策略]
urlcolor: blue
---

目录

<!-- TOC -->

- [1. 逻辑回归](#1-逻辑回归)
	- [1.1 逻辑回归原理](#11-逻辑回归原理)
	- [1.2 逻辑回归损失函数](#12-逻辑回归损失函数)
	- [1.3 逻辑回归优化器](#13-逻辑回归优化器)
	- [1.4 逻辑回归的优缺点](#14-逻辑回归的优缺点)
- [2. 逻辑回归与线性回归区别](#2-逻辑回归与线性回归区别)
- [3. 广义线性模型](#3-广义线性模型)

<!-- /TOC -->

独立研究者 Karanbir Chahal 和 Manraj Singh Grover 与 IBM 的研究者 Kuntal Dey 近日发布了一篇论文，对深度神经网络的分布式训练方法进行了全面系统的总结，其中涉及到训练算法、优化技巧和节点之间的通信方法等。

[论文下载](https://leo4678.github.io/assets/AHitchhikersGuideOnDistributedTrainingOfDNNs.pdf)

[原始论文链接](https://arxiv.org/abs/1810.11787)

## 1. 逻辑回归

### 1.1 逻辑回归原理

逻辑回归首先假定问题域满足Bernoulli分布，模型设计如下：

<html>
<br/>

<img src='/assets/lr梯度求解公式推导.png' style='max-height: 800px; max-width:600px'/>
<br/>

</html>

