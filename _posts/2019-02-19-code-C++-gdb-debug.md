---
layout: post
category: "code"
title: "C++ gdb调试"
tags: [C++]
---

[参考](#https://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/gdb.html)

命令说明：

```
// 编译时，添加-g参数，需要确保编译时debug版本

// 进入gdb工具
gdb tools/caffe-d

// 启动程序 run + 参数 （train -solver=...为参数）
(gdb) r train -solver=../application/weishi/mutil_reader_solver.prototxt

// 设置断点
b 253

// 下一步，碰到函数则进入
s 

// 下一步，碰到函数不进入
n

// 打印变量值
p value
```