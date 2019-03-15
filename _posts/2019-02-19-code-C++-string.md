---
layout: post
category: "code"
title: "C++ 字符串处理技巧"
tags: [C++]
---

目录

<!-- TOC -->

- [1. 字符串声明](#1-字符串声明)
	
- [2. 字符串处理](#2-字符串处理)
	- [2.1 字符串切分](#21-字符串切分)
	- [2.2 字符串长度](#22-字符串长度)
	- [2.3 字符串拼接](#23-字符串拼接)
	- [2.4 为字符串重新分配内存](#24-为字符串重新分配内存)
	- [2.5 字符串转char](#25-字符串转char)

<!-- /TOC -->

## 1. 字符串声明

```c++
//引入头文件
#include <string>

//声明一个空字符串
string s

//拷贝构造函数生成str的复制品
string s(str)

//将字符串str，从stridx的位置开始构造一个字符串
string s(str, stridx)

//将字符串str，从stridx的位置开始且长度最多strlen的部分作为字符串的初值
string s(str, stridx, strlen)

//将cstr C字符串作为s的初值
string s(cstr)

//将chars数组前chars_len个字符作为字符串s的初值
string s(chars, chars_len)

//生成一个字符串，包含num个c字符
string s(num, c)

//以区间beg，end不包含end内的字符作为字符串s的初值
string s(beg, end)

//销毁所有字符，释放内存
s.~string()
```

## 2. 字符串处理

### 2.1 字符串切分

使用boost库

```c++
vector<string> stages;
boost::split(stages, FLAGS_stage, boost::is_any_of(",")); //FLAGS_stage是待切分字符串
```

### 2.2 字符串长度

```c++
//返回当前字符串长度
s.size()
s.length()

//返回当前字符串最多能包含的字符数
s.max_size()
```

### 2.3 字符串拼接

```
//数值型转换为字符串并拼接
FLAGS_gpu = "" + boost::lexical_cast<string>(solver_param.device_id())

//字符串拼接，包括各种数值类型
ostringstream s;
s << "test1 = " << 1; //拼接‘test1 =’与1 
LOG(INFO) << "Using " << s.str();
```

### 2.4 为字符串重新分配内存

```c++
s.reserve(int param)
```

### 2.5 字符串转char

```c++
string s;
s.data(); //返回const char * 字符数组，不添加'\0' 
s.c_str(); //返回const char * 字符数组，添加'\0'

//如需要char *，且确定该函数内部不会对参数做出修改，可直接强制转换
(char *)s.c_str();
```