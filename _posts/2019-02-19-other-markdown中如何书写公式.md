---
layout: post
category: "other"
title: "markdown中如何书写公式"
tags: [markdown公式]
---

目录

<!-- TOC -->

- [1. 在线编写公式](#在线编写公式)
	
- [2. 借助其他插件显示](#借助其他插件显示)
	- [2.1 借助Google Chart服务器](#2.1)
	- [2.2 借助forkosh服务器](#2.2)
	- [2.3 借助MathJax引擎（强烈推荐）](#2.3)

<!-- /TOC -->

## 1. 在线编写公式

在线网站： [https://www.codecogs.com/latex/eqneditor.php](https://www.codecogs.com/latex/eqneditor.php) 可以在该网站编写公式
公式编写规则： https://www.jianshu.com/p/7c34f5099b7e

<html>
<br/>

<img src='../assets/在线编写md数学公式.png' style='max-height: 450px;max-width:750px'/>
<br/>

</html>

## 2. 借助其他插件显示

### 2.1 借助Google Chart服务器

```html
<img src="http://chart.googleapis.com/chart?cht=tx&chl= 在此插入Latex公式" style="border:none;">
ex：
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" style="border:none;">
```

### 2.2 借助forkosh服务器

```html
<img src="http://www.forkosh.com/mathtex.cgi? 在此处插入Latex公式">
ex:
<img src="http://www.forkosh.com/mathtex.cgi? \Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}">
```

### 2.3 借助MathJax引擎（强烈推荐）

在markdown头部添加外部js引用
```html
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
```

在markdown中直接输入公式
- 行间公式 

```
$$公式$$
```

- 行内公式 

```
\\(公式\\)
```
