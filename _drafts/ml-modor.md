---
title: "Modor架构介绍"
urlcolor: blue
---

目录

<!-- TOC -->

- [1. Modor架构](#1-Modor架构)
- [2. 特征配置/算法配置](#2-特征配置/算法配置)
- [3. 数据准备](#3-数据准备)
- [4. 算法](#4-算法)
- [5. 日志](#5-日志)
- [6. 部署](#6-部署)

<!-- /TOC -->

## 1. Modor架构

<html>
<br/>
<img src='modor-framework.png' style='max-height:578px;max-width:690px;'/>
<br/>
</html>

说明：

- Application: 
	- ModelTrain 模型训练任务入口
	- ModelPredict 模型预测任务入口
- DeployUtils:
	- FeatureDeploy 特征出库
	- IndexDeploy 特征编码出库
- TaskConf: 
	- Feature/Algo Conf 模型训练/预测任务特征配置，算法参数配置
- AlgorithmLib: 
	- Model 模型持久化，参数加载，预测接口等
	- AlgorithmFactory 根据算法配置构建算法实例
	- Optimizer
		- Gradient 负责loss、梯度计算
		- Updater 负责模型参数更新
	- Similarity 用户物品相似度算法集
		- Item-CF 基于item的协同过滤算法
	- Rank 排序算法集
		- PairwiseRank-LR 以PairwiseRank为优化目标的LR算法
		- PairwiseRank-FM 以PairwiseRank为优化目标的FM算法
	- Classification 分类算法集
		- LR 逻辑回归
		- FM 因子分解机
		- FTRL 在线学习算法
		- Xgboost 树模型算法
- Core:
	- AlgorithmContext 算法运行环境相关参数，包括算法注册信息、算法使用特征配置信息、缓存等
	- AlgoConf，FeatureConf Bean/Dao 提供读取数据库，获取算法及特征配置信息的接口
	- User/Item/Label Data Bean/Dao 提供从数据仓库读取用户/物品/样本等数据的接口
	- Paramter 模型参数格式，主要是特征编码
	- Util
		- AbtestUtil 离线流量切分
		- CrossFactorCal 交叉特征计算接口
		- DataUtil 特征及样本数据处理接口，主要是MapJoin和SkewJoin
		- MD5Util 加密方法
		- PgBatchUtil Tpg数据库批量插入和查询接口
		- MllibUtil Spark Mllib封装接口，包括矩阵运算、LibSVM文件存储和加载等
	- Process
		- ProcessContext 数据处理流程总控
		- CacheManager 训练数据缓存管理
		- Graph 数据处理DAG图
		- Unit 数据处理图中的处理单元
		- Processor 数据处理单元中的数据计算逻辑
- CommonUtils:
	- DateUtil 日期处理工具，包括日期转换、计算等
	- HashUtil Hash工具，对数据进行hash编码
	- HdfsUtil HDFS读取，存储工具
	- Logger 日志工具
	- Transformer 数据转换工具，包括String转Int、Float、Double，数组转Vetor，数组转LibPoint等
	- SparkAppTemplate Spark程序处理流程模板
	- DBUtil 数据库访问工具，增查删改能力
		- MysqlUtil 访问Mysql
		- TpgUtil 访问Tpg
	- DWUtil 数据仓库访问工具，读写能力
		- TdwUtil 访问Tdw
		- HiveUtil 访问Hive
		- DatabaseTableInfo 数据仓库表信息，包括分区信息、字段、数据格式等

下面通过一个模型训练任务说明一下上述主要组件如何协同工作



## 2. 特征配置/算法配置

## 3. 数据准备

## 4. 算法

## 5. 日志

## 6. 部署

## 7. 其他

### 7.1 采样

### 7.2 召回

### 7.3 评估





