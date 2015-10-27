---
layout: post
title: 理解解释器并动手实现一个极简解释器
description: 用一个简单的示例程序介绍解释器的概念，并使用Python代码实现这个简单的解释器。
category: Python
---

## 写在前面

每当遇到一些新东西后，我就想了解其背后的工作机制。这也导致我这个非CS科班出身的在CS方面一步步越扎越深，不知道这对我个人来说是好是坏。不管怎么样，在博客文章方面，现在打算分两种类型，一种是系统性的撰写文章介绍Pyston和Python（可能也包含其他Python实现）的内部工作方式，另一种是将学习这些源码过程中掌握的知识点也整理成离散的文章发布。这两种类型的文章都来自于开发Pyston过程中所学到的知识。在这里记录下来，方便自己查阅，也希望能对读者有所帮助，毕竟，如果我能介绍清楚了，那么我也掌握了。

这一篇介绍解释器。因为对于Pyston、JavaScript V8这样的项目，解释器是其三层（或四层）架构当中的第一层。所以文章也是按照编译器实现的顺序一步步撰写的（三层/四层架构是V8和Pyston采用的架构，看我以后能不能写文章填上这个坑吧）。对于CPython来说，解释器是其编译链的第二步。

## 解释器到底是什么？

许多编译原理的书籍或课程中都会说到：“程序语言的执行主要分为两种方式：编译执行和解释执行。”

* **编译执行**是将编程语言经过词法处理、语法处理、语义分析等一系列步骤，翻译成机器代码。然后执行编译后的机器代码。
* **解释执行**是直接处理并运行源程序，不先把源程序翻译成机器语言。
	“解释器直接利用用户提供的输入执行源程序中制定的操作。”——龙书正文第一页
	“解释器将一种可执行规格作为输入，产生的输出是执行该规格的结果。”——《编译器设计（第二版）》第二页

> 注：这篇文章侧重于解释器，并假定读者已经基本掌握了编译方面的相关基本知识。

就我个人而言，编译器的定义并不难懂。但根据前面列举出来的对解释器的定义，我还是很久都没明白解释器到底是什么，只知道解释器不用将源语言翻译成机器语言就能运行程序。

在真正接触了一些解释器的源码后。我才对解释器有了**基本的**的认识。

解释器是在宿主语言中对源语言进行处理，除此之外不接触底层的内容。解释器同样包含词法、语法、语义分析这些步骤。假设我们用C++编写一个针对Python源码的解释器。那么C++就是宿主语言，而Python就是源语言。在这种情况下，首先依然需要用C++编写词法、语法、语义分析程序。

以 `3 + 6` 这个表达式为例，解析器需要处理以下几个步骤：

**解析**：

`3 + 6` 会通过词法等过程解析成

`<Integer, '3'>` 、`<Operator, '+'>` 、`<Integer, '6'>`

**对象表示**：

编译器需要继续生成机器代码，而解析器会直接在宿主语言中执行。比如将`<Integer, '3'>` 和 `<Integer, '6'>` 在C++中分别用 `long` 对象存储（CPython和Pyston中的确是这么做的，当然，会声明一个对象来持有，比如CPython中使用`PyIntObject`，而Pyston中使用`BoxedInt`）。并用一个函数 `intAdd` 表示作用于这个两个整数的 `<Operator, '+'>` 操作。

**处理**：

在 `intAdd` 中对 `IntObject(3)` 和 `IntObject(6)` 中进行处理，实际上就是直接在C++中执行 `3 + 6` ，得到9，存储在新的 `IntObject` 中，这就是结果。 若要显示在屏幕上，则把结果转成字符串形式。在CPython或Pyston中是通过 `intRepr` 这样的函数完成的。

> 说明：这里的伪代码的命名只是用来说明，并不是实际的函数名称。比如CPython中，是通过`int_add`完成针对整数的加法运算的，而Pyston中是通过`intAdd`完成的。`IntObject`也是抽象的表示。

从这三步可以看出，所有的操作都是在C++中执行，没有接触到机器语言。在命令行中，如果输入`3 + 6`，会回显`9`，如下所示：

```
>>> 3 + 6
9
```

这个`9`并不是真正的结果，而只是结果的字符串表达形式。

希望通过这段描述能让读者从另一个角度对解释器有所理解。

但上面仅仅是解释器的一种工作方式。解释器一般有下面两种方式：

* 直接解释源码本身，如Shell语言，本文中这个简单的示例。
* 先由编译器将源码编译成字节码，解释器解释执行编译后的字节码（CPython就是这么干的）。

如果再深入，就会涉及到JIT方面了，而JIT又更接近编译器方面，所以我把解释器的定义就划到这。

> **杂谈**：CPython的交互式命令行使用“>>>”三个箭头作为提示符，PyPy使用四个，Pyston的开发者一看，总不能用五个吧，于是退而秋其次，使用两个箭头作为命令行提示符。

## 简单解释器的实现

为了加深理解，这里为简单的加减法表达式实现一个简单的解释器。宿主语言是Python，源语言是简单的由算术表达式，如下所示：

```
2 + 3
42 + 42
```

这款解释器在[参考资料1](#ref1)的基础上进行修改，添加了一点内容，模拟CPython和Pyston中的一些元素，方便读者今后能理解Pyston中完成的方式。

待解析的表达式为`2 + 3`、`42 - 21`这样的形式，由于这里主要目的是为了介绍解释器的概念，所以将表达式限定为必须是`数字-空格-运算符-空格-数字`这种形式，以省去词法分析等步骤的代码，且目前只能处理单行二元运算的加法和减法。

首先，程序读取一行输入，将其解析为三个Token，如`2 + 3`解析成：
```
<INT, 2>
<OP_PLUS, '+'>
<INT, 3>
```
完成这个任务的代码为：

{% highlight python linenos=True%}
INT, OP_PLUS, OP_SUB = 'INT', 'OP_PLUS', 'OP_SUB'

class Token(object):
    def __init__(self, type, value):
        self.value = value
        self.type = type

class Interpreter(object):
    def __init__(self, expr):
        self.expr = expr
        self.tokens = []

    def parse(self):
        expr = self.expr
        units = expr.split()
        for unit in units:
            try:
                result = int(unit)
                token = Token(INT, result)
                self.tokens.append(token)
            except ValueError:
                pass
            if unit == '+':
                token = Token(OP_PLUS, unit)
                self.tokens.append(token)
            if unit == '-':
                token = Token(OP_SUB, unit)
                self.tokens.append(token)
{% endhighlight %}

这里的代码进行了简化，没有使用正则或状态机，直接假定输入的表达式中的每个词法元素由空格隔开。然后分别生成相应的Token。

接着将`<INT, 2>`这样的Token转成`IntObject(2)`，模拟CPython和Pyston中的处理方式，CPython中会用`PyIntObject`持有，Pyston会用`BoxedInt`持有。然后根据运算符的类型，选择`IntObject`中对应的操作函数（CPython中为`tp_slot`）：

{% highlight python linenos=True%}
class IntObject(object):
    def __init__(self, value):
        self.value = value

    def add(self, other):
        return IntObject(self.value + other.value)

    def sub(self, other):
        return IntObject(self.value - other.value)

    def __str__(self):
        return str(self.value)

# class Interpreter(object):
    # 位于Interpreter类中
    def eval(self):
        self.parse()
        left = self.tokens[0]
        right = self.tokens[2]
        lhs = IntObject(left.value)
        rhs = IntObject(right.value)

        op = self.tokens[1]
        if op.type == OP_PLUS:
            return lhs.add(rhs).value
        elif op.type == OP_SUB:
            return lhs.sub(rhs).value
{% endhighlight %}

最后使用一个主函数运行这个解释器：

{% highlight python linenos=True%}
def run_repl():
    while True:
        try:
            expr = raw_input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not expr:
            continue
        repl = Interpreter(expr)
        result = repl.eval()
        print(result)

if __name__ == '__main__':
    run_repl()
{% endhighlight %}

这里使用Python代码编写，简化了代码量。如果使用C/C++编写，在Linux下可以使用GNU readline库。

## 总结

上面的代码片段中包含了这个简单解释器的全部内容。但代码并不是按顺序贴出来的。如果读者有兴趣，可以自己将这些代码组合起来，加深自己的理解。

这里的代码在照顾概念的同时，最大程度进行了简化。比如要求输入的表达式必须用空格划分词法单元；没有错误检查；直接将表达式的各个词法单元存入到一个列表中；使用硬编码的方式解析表达式。

希望这些内容已经能清楚的介绍什么是解释器。后续文章将介绍其他概念，如果有时间的话，也可能像[参考资料1](#ref1)那样补充成一个系列，专门介绍如何实现一个完整的解释器。

## 参考资料

1. <span id="ref1">《构建一个简单的解释器》</span>，[中文版](http://blog.jobbole.com/88152/)，[英文版](http://ruslanspivak.com/lsbasi-part1/)。
2. [《龙书》](http://www.amazon.cn/%E7%BC%96%E8%AF%91%E5%8E%9F%E7%90%86-Alfred-V-Aho/dp/B001NGO85I/)
3. [《编译器设计（第二版)》](http://www.amazon.cn/%E7%BC%96%E8%AF%91%E5%99%A8%E8%AE%BE%E8%AE%A1-%E5%BA%93%E4%BC%AF/dp/B00ASOR6N2/)
