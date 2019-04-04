---
title: "Modor架构介绍"
urlcolor: blue
---

目录

<!-- TOC -->

- [1. Modor架构](#1-Modor架构)
- [2. Conf/配置](#2-Conf/配置)
- [3. Application/应用层](#3-Application/应用层)
- [4. DeployUtils/部署层](#4-DeployUtils/部署层)
- [5. AlgorithmLib/算法层](#5-AlgorithmLib/算法层)
- [6. Core/核心层](#6-Core/核心层)
- [7. CommonUtils/通用工具层](#6-CommonUtils/通用工具层)
- [8. 各组件调用详解](#6-各组件调用详解)

<!-- /TOC -->

## 1. Modor架构

<html>
<br/>
<img src='modor-framework.png' style='max-height:578px;max-width:690px;'/>
<br/>
</html>

说明：

- Conf: 
	- Feature/Algo Conf 数据库中模型训练/预测任务特征配置，算法参数配置信息
- Application: 
	- ModelTrain 模型训练任务入口
	- ModelPredict 模型预测任务入口
- DeployUtils:
	- FeatureDeploy 特征出库
	- IndexDeploy 特征编码出库
- AlgorithmLib: 
	- Model 模型持久化，参数加载，预测接口等
	- AlgorithmFactory 根据算法配置构建算法实例
	- Optimizer 优化器
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

下面通过一个**模型训练的时序图**说明上述**主要组件如何协同工作**

<html>
<br/>
<img src='modor算法训练时序图.png' style='max-height:578px;max-width:690px;'/>
<br/>
</html>

## 2. Conf/配置

该模块指的是在[微信支付/数据平台/算法平台/特征和算法配置](http://wxpay.oa.com/dataplatform/index/data?page=7_0)进行相关配置。它的作用是统一管理算法训练时所需要的信息，包括**特征注册、算法基础信息注册、算法使用特征信息**。

## 3. Application/应用层

## 4. DeployUtils/部署层

## 5. AlgorithmLib/算法层

## 6. Core/核心层

## 7. CommonUtils/通用工具层

## 8. 各组件调用详解





