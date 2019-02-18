---
layout: post
category: "dl"
title: "deepr数据预处理"
tags: [数据预处理, 标准化/Standardization, 归一化/正则化/Normalization, one-hot, muti-hot]
urlcolor: blue
---

# 标准样本准备
**目录**
[TOC]
## 1. 目标
标准样本准备的目的是为RCaffe模型训练提供标准格式的数据集。以下是详细步骤：
### 1.1 TDW样本数据准备
```
CREATE TABLE IF NOT EXISTS table_name(
    ftime BIGINT COMMENT '组特征时间',
    sample_id BIGINT COMMENT '样本i', --每条样本对一个id,便于反查数据问题
    sample_weight FLOAT COMMENT '样本权重', --该条样本整体分布中所占比重
    labels STRING COMMENT 'labels', --多个拟合目标以英文逗号隔开
    gf1_type STRING COMMENT '组特征类型', --目前仅支持两种组特征格式(一是CVS[高稀疏] key-value类型)
    gf1_values STRING COMMENT '组特征内容',
    gf2_type STRING COMMENT '组特征类型', --目前仅支持两种组特征格式(一是NORMAL[稠密型] dense 数组类型)
    gf2_values STRING COMMENT '组特征内容'
)
COMMENT 'table描述信息'
PARTITION BY LIST( ftime )
(
    PARTITION default
)
STORED AS RCFILE COMPRESS;
```
**说明：
1.固定非空字段：ftime, sample_id, sample_weight, labels
2.可重复字段：gf[i]_type, gf[i]_values 这两个字段配对出现且必须连续，type在前，value在后。**
gf[i]_type只有两种值'CVS','NORMAL'
gf[i]_values 
_CVS对应示例 age\*36_40:1.0|oper_time\*week_day:4|oper_time\*hour:9, 该value中包含三个key-value对，分别是age\*36_40:1.0，oper_time\*week_day:4，oper_time\*hour:9，多个key-value对以‘|’拼接，每一个key-value中‘:’前为key值，后为value值，在key中使用‘\*’来连接primary_key和secondary_key
NORMAL对应示例 0.1|1.2|3.0|0|0|1.0，所有样本该特征组的尺寸都必须保证一致_
**3.字段名都可个性化重命名，但顺序必须与示例保持一致**
下图是存储在tdw的一个标准RCaffe样本数据表
![image.png](/uploads/66690AF88DA64B459397BEA40B129601/image.png)

### 1.2 样本数据标准化处理
* 组特征编码
处理CVS类型组特征，将原始key映射为索引id
* 脏数据过滤
对样本表中的每一条样本进行检测，目前检测内容为：labels字段值检测(检测多目标数目和label值是否合法)，gf[i]_type检测(检测组特征是否合法)，gf[i]_value检测(检测组特征值是否合法，尺寸是否合法)
* copy到HDFS
由于RCaffe暂未提供直接读取tdw库表文件的功能，所以需要将tdw样本数据落盘到hdfs上
* ProtoBuf转换
将落盘到hdfs上的样本转换成PB格式
* lmdb转换
将PB格式转换为RCaffe要求的lmdb格式

## 2. 参考示例
以下以实际示例为例，一步步来说明数据准备如何来做。先介绍一下该示例的背景，该示例的目的是为了建立用户和物品的稠密低维空间表达，进而使用该稠密低维表达来计算用户和物品的匹配程度。它分了两组组特征，一组为用户侧特征，另一组为物品侧特征。
> ![rcaffe_rnn_desc.png](/uploads/CE14B617833E4485942F0CD62F532082/rcaffe_rnn_desc.png)

### 2.1 样本数据准备
该部分所用流程仅供参考，如有更加便捷快速的方法欢迎交流。
**2.1.1 构建样本、特征**
该步骤可参考“神盾离线”样本、特征规范
**2.1.2 样本-单特征拼接**
该步骤将样本与该样本匹配到的特征join

>示例python脚本 [weishi_rcaffe_nn_recall_gf_prepare.py](/uploads/95C32436FDCB4AC48A54FB8918DA46AF/weishi_rcaffe_nn_recall_gf_prepare.py)

```
--建立临时中间表
CREATE TABLE IF NOT EXISTS tb_name(
    ftime BIGINT COMMENT '时间',
    user_id STRING COMMENT '用户id',
    item_id STRING COMMENT '物品id',
    sample_weight FLOAT COMMENT '样本权重',
    label STRING COMMENT 'label',
    gf_ix TINYINT COMMENT '组特征索引',
    feature STRING COMMENT '特征值'
)
COMMENT ''
PARTITION BY LIST( ftime )
SUBPARTITION BY LIST(gf_ix)
(
    SUBPARTITION sp_gf_0 VALUES IN ('0'),
    SUBPARTITION sp_gf_1 VALUES IN ('1'),
    SUBPARTITION default
)
(
    PARTITION default
)
STORED AS RCFILE COMPRESS;

--插入数据
--用户播放过物品最近top100
INSERT TABLE %(tb_name)s --表名       
SELECT    
        %(date)s, --时间
        user_id, item_id, 1.0, label,
        0, --0特征组id
        concat(concat(concat(concat(primary_key, '*'), secondary_key), ':'), value) AS feature
    FROM
        (
            SELECT
                    DISTINCT user_id, item_id, label
                FROM
                    %(sample_dbtb_name)s PARTITION(p_%(date)s) t --sample_dbtb_name为样本库表名
        ) l1        
    JOIN
        (
            SELECT 
                    owner, primary_key, secondary_key, value
                FROM 
                    (
                        SELECT
                                owner, primary_key, secondary_key, value, row_number() over (PARTITION by owner order by ftime, secondary_key desc) rank
                            FROM
                                hlw::weishi_rcaffe_nn_recall_user_feature
                            WHERE
                                ftime BETWEEN %(three_day_late)s AND %(date)s
                                AND primary_key = 'user_complete_play_item'
                    ) t
                WHERE
                    t.rank <= 100 --用户最近播放100个物品
        ) l2
    ON
        l1.user_id = l2.owner

--用户点赞category偏好程度
INSERT TABLE %(tb_name)s        
SELECT    
        %(date)s,
        user_id, item_id, 1.0, label,
        0, --0特征组id
        concat(concat(concat(concat(primary_key, '*'), secondary_key), ':'), value) AS feature
    FROM
        (
            SELECT
                    DISTINCT user_id, item_id, label
                FROM
                    %(sample_dbtb_name)s PARTITION(p_%(date)s) t
        ) l1        
    JOIN
        (
            SELECT
                    owner, primary_key, secondary_key, sum(value) AS value
                FROM
                    hlw::weishi_rcaffe_nn_recall_user_feature
                WHERE
                    ftime BETWEEN %(seven_day_late)s AND %(date)s
                    AND primary_key = 'user_like_category_degree'
                GROUP BY
                    owner, primary_key, secondary_key
        ) l2
    ON
        l1.user_id = l2.owner
--其他用户侧特征省略

--物品id特征
INSERT TABLE %(tb_name)s        
SELECT    
        %(date)s,
        user_id, item_id, 1.0, label, 
        1, --1特征组id
        concat(concat('item_id*', item_id), ':1.0') AS feature
    FROM
        %(sample_dbtb_name)s PARTITION(p_%(date)s) t
--其他物品侧特征省略
```
tdw中数据表现格式如下图所示：
![image.png](/uploads/67650BCC9DF34199B46DDA8CB9080F49/image.png)

**2.1.3 样本-单特征转为组特征**
上一步将样本和单个特征进行了拼接，这里将单特正转为特征组
>示例python脚本 [weishi_rcaffe_nn_recall_sample_join_feature_v1.py](/uploads/89C6DC1BA21B4715BC923AE6AEEC0FE8/weishi_rcaffe_nn_recall_sample_join_feature_v1.py)

```
--建表
CREATE TABLE IF NOT EXISTS %(tb_name)s(
    ftime BIGINT COMMENT '组特征时间',
    sample_id BIGINT COMMENT '样本id',
    sample_weight FLOAT COMMENT '样本权重',
    labels STRING COMMENT 'labels,多目标',
    gf1_type STRING COMMENT '组特征类型', --CVS[高稀疏] user feature
    gf1_values STRING COMMENT '组特征内容',
    gf2_type STRING COMMENT '组特征类型', --CVS[高稀疏] item feature
    gf2_values STRING COMMENT '组特征内容'
)
COMMENT 'Rcaffe-微视-召回WD-NN-样本v1'
PARTITION BY LIST( ftime )
(
    PARTITION default
)
STORED AS RCFILE COMPRESS;

--插入特征组数据
INSERT TABLE %(tb_name)s
SELECT
        %(date)s,
        row_number() over(order by user_id, item_id, label desc) sample_id, 
        sample_weight,
        label,
        'CVS',
        gf0_values,
        'CVS',
        gf1_values
    FROM
        (
            SELECT
                    user_id, item_id, label, 
                    max(sample_weight) AS sample_weight,
                    max(IF (gf_ix = 0, features, '')) AS gf0_values,
                    max(IF (gf_ix = 1, features, '')) AS gf1_values
                FROM
                    (
                        SELECT
                                user_id, item_id, label, gf_ix, max(sample_weight) AS sample_weight, wm_concat(distinct feature, '|') AS features
                            FROM
                                %(gf_db_tb_name)s PARTITION(p_%(date)s) t
                            GROUP BY
                                user_id, item_id, label, gf_ix
                    )
                GROUP BY
                    user_id, item_id, label    
        )
```
### 2.2 标准化处理-编码，过滤，落盘
这里使用Spark来完成，需在洛子或tesla上配置Spark计算任务。
> 配置任务可参考：http://lz.oa.com/action_manage?TaskID=20180919211518214
任务运行jar包[DataTools-Application-10-jar-with-dependencies.jar](/uploads/B4F778FC337040D3BF7B7DD5F900FA80/DataTools-Application-10-jar-with-dependencies.jar)

参数说明
```
ftime=${YYYYMMDD} #指明tdw中样本表时间分区
task_type=tdw_sample_to_hdfs #指明任务类型，目前仅支持'tdw_sample_to_hdfs'，将tdw样本数据落盘到hdfs
db_name=hlw #tdw样本库名
tb_name=weishi_rcaffe_nn_recall_sample_join_feature_v1 #tdw样本表名
label_size=1 #拟合目标数目，与样本中labels个数对应
gf_size=2 #组特征数目
gfs_info=CVS-0-0;NORMAL-4-2 #组特征类型，尺寸大小，'CVS-0-0'表示组特征类型为CVS，0-0无实际意义，'NORMAL-4-2'表示组特征类型为NORMAL，4表示四组特征，每组特征有两个value值来表示，其组特征配置信息要与gf_size参数一致
```

任务执行的中间信息可查看日志，如找不到日志也可从hdfs上查看日志，方法如下：
```
#登陆任意一台能够连接到 hdfs://tl-if-nn-tdw.tencent-distribute.com:54310 集群的机器，如：10.222.0.92(注：需要先登录跳板机)
ssh name@10.222.0.92
```

查看日志
```
#命令 hadoop_tdw_if  -cat /stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/log/${YYYYMMDD}/${sample_db_name}__${sample_tb_name}.log
[davewli@bigdata-hive ~]$ hadoop_tdw_if  -cat /stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/log/20180928/hlw__weishi_rcaffe_nn_recall_sample_join_feature_v1.log
[INFO] - 2018-09-29 11:36:57 - =========Begin=========
[INFO] - 2018-09-29 11:36:57 - *** Command Args *** : 
[INFO] - 2018-09-29 11:36:57 - task_type=tdw_sample_to_hdfs
[INFO] - 2018-09-29 11:36:57 - tb_name=weishi_rcaffe_nn_recall_sample_join_feature_v1
[INFO] - 2018-09-29 11:36:57 - gfs_info=CVS-0-0;CVS-0-0
[INFO] - 2018-09-29 11:36:57 - label_size=1
[INFO] - 2018-09-29 11:36:57 - gf_size=2
[INFO] - 2018-09-29 11:36:57 - db_name=hlw
[INFO] - 2018-09-29 11:36:57 - ftime=20180928
[INFO] - 2018-09-29 11:36:57 - ********************
[INFO] - 2018-09-29 11:37:17 - -------RCaffe Data Prepare Processing Begin -------
[INFO] - 2018-09-29 11:37:20 - From hlw::weishi_rcaffe_nn_recall_sample_join_feature_v1 p_20180928 get 1891266 records
[INFO] - 2018-09-29 11:40:43 - Check sample, get 1891266 valid sample
[INFO] - 2018-09-29 11:46:23 - Successfully save RcaffeSampleFeatureEncoding to [ hlw::rcaffe_sample_feature_encoding PARTITION(p_20180928, sp_hlw__weishi_rcaffe_nn_recall_sample_join_feature_v1)] 
[INFO] - 2018-09-29 11:46:23 - train sample total 1513427
[INFO] - 2018-09-29 11:48:10 - test sample total 377839
[INFO] - 2018-09-29 11:50:21 - HDFS rm Path[/stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/sample/20180928/hlw__weishi_rcaffe_nn_recall_sample_join_feature_v1_train] ... false
[INFO] - 2018-09-29 11:50:21 - HDFS rm Path[/stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/sample/20180928/hlw__weishi_rcaffe_nn_recall_sample_join_feature_v1_test] ... false
[INFO] - 2018-09-29 11:50:21 - Save encoding&check train sample data to hdfs://tl-if-nn-tdw.tencent-distribute.com:54310/stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/sample/20180928/hlw__weishi_rcaffe_nn_recall_sample_join_feature_v1_train
[INFO] - 2018-09-29 11:53:26 - Save encoding&check test sample data to hdfs://tl-if-nn-tdw.tencent-distribute.com:54310/stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/sample/20180928/hlw__weishi_rcaffe_nn_recall_sample_join_feature_v1_test
[INFO] - 2018-09-29 11:56:14 - -------Finished-------
[INFO] - 2018-09-29 11:56:14 - =========End=========
[INFO] - 2018-09-29 11:56:14 - Saving Logs:  to Path[/stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/log/20180928/hlw__weishi_rcaffe_nn_recall_sample_join_feature_v1.log]
[INFO] - 2018-09-29 11:56:14 - HDFS rm Path[/stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/log/20180928/hlw__weishi_rcaffe_nn_recall_sample_join_feature_v1.log] ... false
```

### 2.3 标准化处理-PB，lmdb转换
2.3.1 从hdfs上下载样本到本地磁盘 
```
#登陆可连接到hdfs系统的机器之后，使用如下命令拉取文件到本地磁盘
#hadoop_tdw_if  -getmerge /stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/sample/${YYYYMMDD}/${sample_db_name}__${sample_tb_name} ${local_sample_name}
[davewli@bigdata-hive ~]$ hadoop_tdw_if  -getmerge /stage/outface/sng/g_sng_im_g_sng_own_tdw_shield/rcaffe/sample/20180928/hlw__weishi_rcaffe_nn_recall_sample_join_feature_v1_test test.sample
```
**注意：这里需要拉取测试集和训练集，日志中带有test标识的为测试集，train标识的为训练集**

2.3.2 PB-lmdb转换
> 工具源码 [RCaffeDataTool.zip](/uploads/6BBDCA1A40BD4D78B2B1AAA5092FBA3C/RCaffeDataTool.zip)

```
#下载上述工具源码，并解压，执行如下命令
#命令 python ./src/rcaffe/tools/DataTransfrom.py -tt transfrom -s ${local_sample_file} -t ${lmdb_target}
#参数说明
#    -tt :task_type 转换任务 or 读取lmdb任务
#    -s :source_path 源文件路径
#    -t :target_lmdb_path lmdb目标路径

python ./src/rcaffe/tools/DataTransfrom.py -tt transfrom -s /data/home/davewli/DeepR/SampleData/hlw__rcaffe_sample_weishi_cm_train_data.20180830.sample -t /data/home/davewli/DeepR/RCaffe_LMDB/RCaffe_weishi_cm_train.lmdb
```
**注意：分别对测试集和训练集执行转换操作**

`至此，准备了***_test.lmdb和***_train.lmdb，RCaffe模型训练所需要的数据就准备好了`
```
drwxr-xr-x 2 davewli davewli 4096 Sep 28 14:51 RCaffe_weishi_wide_recall_v3_test.lmdb
drwxr-xr-x 2 davewli davewli 4096 Sep 28 14:52 RCaffe_weishi_wide_recall_v3_train.lmdb
```

如有问题，可联系davewli，欢迎交流
