---
layout: post
title: Pyston：理想的Python实现
description: 对比CPython、PyPy、Pyston的技术背景，不难发现Pysthon是理想的Python实现。即支持Python C API，又能大幅度提升执行效率。
category: Python
---

## 简介

Pyston是采用JIT技术的Python实现，目前刚刚发布0.4版，[这篇文章](http://blog.jobbole.com/65414/)对其做了介绍。其中也解释了为什么Dropbox创建一个新的Python实现。Pyston目前仍无法在生产环境中使用，其成熟度仍无法与CPython等其他Python实现相比。

同时由于有Unladen Swallow这个先例在前面，所以这里还要加一个前提：只要Pyston坚持下去，那么绝对是各种Python实现中的佼佼者，因为Pyston几乎是理想的Python实现。

为什么这么说呢，因为对Python用户而言，最大的需求在于性能和已有模块的支持这两方面。CPython作为标准的Python实现，现有的Python扩展就是为它量身定做的，但其开发效率堪忧。而PyPy是铁了心要追求效率，对扩展模块的支持比较匮乏。IronPython和Jython则分别在各自的领域发挥作用，不像PyPy是争夺“大统”的。

**更新**：更新了[一篇文章](http://daetalus.github.io/2015/10/26/python_implementation_comparison/)，其中简要介绍了几种常见的Python的实现

PyPy是一款优秀的Python实现（本文中的PyPy是指狭义的Python实现，不包括RPython工具链）,在性能方面作出了很大的提升。但有优点的同时，也有一些根本性的问题阻止她流行开来。

之前有一篇文章介绍过PyPy（[《为什么PyPy是Python的未来》](http://blog.jobbole.com/39757/)）。其中介绍了Python和PyPy的诸多优点，文中大力推介PyPy，但有些地方也有失公允，下面会提到。

## 扩展模块的支持

PyPy最为人诟病的是其缺乏对C扩展模块的完善支持，这也是PyPy没有得到大规模应用的重要原因。PyPy对扩展的支持主要有两种，一种是用RPython重写，集成到PyPy中。另一种是采用cffi模块将C/C++编写的库引入到PyPy中。但这样无法利用已有的为CPython编写的扩展。PyPy官方建议采用cffi的方式支持扩展模块。

虽然PyPy模拟了一些Python的C API，但通过这些模拟的API来支持扩展模块，比在CPython中运行的还慢。

而Pyston从设计之初就考虑到要原生支持已有的Python C API。因此这方面不是问题，目前Pyston已经能支持多个使用Python C API的扩展。不过为此做的妥协就是，Pyston采用的是保守的Mark-Sweep（标记清除法）的垃圾回收。而PyPy可以通过参数选择不同的GC模式。

在《为什么PyPy是Python的未来》这篇文章中，有些地方有失公允。比如PyPy不支持使用CPython C API编写的扩展，然后文中就大力鼓吹C API不好，不要用C API。但文中并没有给出强有力的论据来支持这个观点。相反，我认为C API还是有非常可取的地方。

举个例子来说，OpenCV的Python封装中，就是采用C API做到C++编写的OpenCV与Python的无缝结合。具体来说，在C++的层面调用CPython的C API，以及NumPy提供的C API，在进入Python层面之前，就将OpenCV中的Mat（UMat）转换成NumPy中的`ndarry`。在Python中完全以Pythonic的方式使用OpenCV的API。当然，这样需要手动编写一个OpenCV的API转换程序，不过这是OpenCV开发者经过尝试和权衡作出的决定，在此之前，他们也尝试过使用SWIG。而PyPy推崇的cffi是无法做到这么细腻的，更不用说提供全Pythonic的接口了。

事实上，无法指望第三方库同时使用两套接口方案来支持不同的Python实现。这种情况下，PyPy只能自己去重新实现或由爱好者使用cffi重新编写扩展模块。注意，前者是重新实现，就比如NumPy，他们重新实现了一个NumPypy。这个代价简直是无法想象的。

## JIT支持

Pyston采用的是method-at-a-time方式的JIT，而PyPy采用的是tracing JIT（后续会尽量补充一篇文章来介绍这两者）。正如Pyston开发者所说，如果PyPy找到了最快的路径，那么Pyston是无法在性能上击败PyPy的。因为tracing JIT会内联所有内容，甚至能做的更好，如只内联实际执行的内容。这样，当每次都是以相同的路径执行时，只需走固定的路径，无需进行额外的分支选择、检查等工作。同时也将那些不会执行的路径也即时编译掉。但这并不意味着Pyston会比PyPy慢，因为method-at-a-time JIT也有自己的优势。比如，如果一个方法有多个路径，如果并不是总是确保每次都只执行一条路径，则tracing JIT必须要回退到解释器中，而method-at-a-time则不用。

Method-at-a-time与tracing JIT这两种实现各有伯仲。JavaScript V8就是method-at-a-time的JIT实现。而LuaJIT采用的是tracing JIT。这两者都算是成功的范例。

## 总结与展望

Pyston有完善的C API支持，同时又有在理论上不输于PyPy的速度、先进的架构和设计。因此Pyston完全是理想的Python实现。

不过Pyston刚刚发布0.4，还有很长的一段路要走。目前Pyston只支持X86-64架构的Linux。尚不支持32位，更不用说Windows和Mac了。同时Pyston的GC目前也很不稳定。最后，Pyston目前是针对Python 2进行开发中。Python 3的支持尚未提上日程。

但随着时间的推移，只要Pyston坚持下去，这些问题是可以解决的。
