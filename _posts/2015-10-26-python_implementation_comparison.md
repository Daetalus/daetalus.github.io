---
layout: post
title : 各种Python实现的简单介绍与比较
description: Python有多种实现，如最常用的CPython，以速度著称的PyPy，构建与JVM或.NET之上的Jython和IronPython等。这篇文章简要介绍各个实现。
category: Python
---

当谈到Python时，一般指的是CPython。但Python实际上是一门语言规范，只是定义了Python这门语言应该具备哪些语言要素，应当能完成什么样的任务。这种语言规范可以用不同的方式实现，可以用C实现，也可以用C++、Java、C#、JavaScript，甚至使用Python自己实现。这篇文章就是简要介绍并比较不同的Python实现，并且今后还会不断的扩充。

## CPython

[CPython](www.python.org)是标准Python，也是其他Python编译器的参考实现。通常提到“Python”一词，都是指CPython。CPython由C编写，将Python源码编译成CPython字节码，由虚拟机解释执行。没有用到JIT等技术，垃圾回收方面采用的是引用计数。

所以当有人问道Python是解释执行还是编译执行，可以这样回答：Python（CPython）将Python源码编译成CPython字节码，再由虚拟机解释执行这些字节码。

如果需要广泛用到C编写的第三方扩展，或让大多数用户都能直接使用你的Python代码，那么还是使用CPython吧。

## Jython

[Jython](https://hg.python.org/jython)在JVM上实现的Python，由Java编写。Jython将Python源码编译成JVM字节码，由JVM执行对应的字节码。因此能很好的与JVM集成，比如利用JVM的垃圾回收和JIT，直接导入并调用JVM上其他语言编写的库和函数。

对于想在JVM上使用Python简化工作流程，或者出于某些原因需要在Python语言中使用Java的相关代码，同时无需用到太多CPython扩展的用户来说，极力推荐Jython。

## IronPython

[IronPython](http://ironpython.net/)与Jython类似，所不同的是IronPython在CLR上实现了Python，即面向.NET平台，由C#编写。IronPython将源码编译成**TODO CLR**，同样能很好的与.NET平台集成。即与Jython相同，可以利用.NET框架的JIT、垃圾回收等功能，能导入并调用.NET上其他语言编写的库和函数。IronPython默认使用Unicode字符串。

另外，Python Tools for Visual Studio可以将CPython和IronPython无缝集成进VS中。如果仅需要在Windows上开发较大的Python项目。条件允许的情况下，IronPython是个不错的选择。

## PyPy

这里说的PyPy是指使用RPython实现，利用Tracing JIT技术实现的Python，而不是RPython工具链。PyPy可以选择多种垃圾回收方式，如标记清除、标记压缩、分代等。

想对于CPython，PyPy的性能提升非常明显，但对第三方模块的支持真心是弱爆了。比如无法很好的支持使用CPython的C API编写的扩展，完全不支持使用SWIG、SIP等工具编写的扩展。就连NumPy，也要在编译器的层面上从头实现。即使实现了，也只能在Python层面中使用，无法供其他第三方模块在非Python环境中使用。关于PyPy，后续会尝试用一篇完整的文章来介绍。不过我的[这一篇文章](http://daetalus.github.io/2015/10/25/pyston_ideal_python/)中对PyPy和下面的Pyston有更详细的描述。

## Pyston

Pyston由Dropbox开发，使用C++11编写，采用Method-at-a-time-JIT和Mark Sweep——Stop the World的GC技术。Pyston使用类似JavaScript V8那样的多层编译，其中也用到了LLVM来优化代码。Pyston正在发展中，还不成熟。但其前景是非常值得看好的（如果没像Google的Unladen Swallow那样死掉的话。话说，Google的东西现在是越来越不敢用了，不是他们的东西不好，是怕用着用着，他们就关掉了）。

## 总结

这里介绍了主要（其实是我接触过的，^\_^）的几款Python实现，这几款Python实现可以满足大部分需要。而略过了几款，如Cython、Brython、RubyPython等。其实Cython还是挺有用的，不过现在接触的不多，不敢多写，看后面能不能抽时间补上。而Brython、RubyPython，个人感觉完全可以用JS或Ruby，没必要在一种动态语言的环境中再使用另一种动态语言。
