---
layout: post
title: Data Type in Python
---

## Value in Box v.s. Binding Name on Object

In C programming language, when we assign a value to a variable, it actually create a block of memory space so that it can hold the value for that variable. So

```python
int a = 1;
```

is like to put the value in a box(memory space) with the variable name as following
![enter image description here](https://lh3.googleusercontent.com/H_VkNqeuwnFsQJimdifOAA76wLklBVl-Dl3JzrhTyfU=s0 "a1box.png")

If you change the value of the variable, then the new value will be put in the box, so 

```python
int a = 2;
```

will result in
![enter image description here](https://lh3.googleusercontent.com/27oN3JtVfNiu4V7QQI58nslrPmHqE6yRVom2NynGkNs=s0 "a2box.png")

However, in Python, the variable works in a quite different way. When we write```int a = 1``` in Python, it's wrong to say that ```a``` "contains" a ```1``` object[^Note1]. Rather, ```a``` is the **name** with **binding** to the **object** ```10```. **Instead of variables, Python has name and bindings**, as shown below:

![enter image description here](https://lh3.googleusercontent.com/JUMJNSI1tjyiyWr-HEGDbNX0ZqQAjjUlVa8gtU8plR8=s0 "a1tag.png")

Python just binds the name ```a``` to the object ```1```, when we change the value variable, we actually binds the name ```a``` to another object, so```a = 2```means
![enter image description here](https://lh3.googleusercontent.com/NsG7IiHV3g4zvufxgQTndx_Wb5Wq9pxI0OA7fdZHVms=s0 "a2tag.png")

## Mutable Type v.s. Immutable Type

So what will happen if we assign a variable to another?

```python
a = 2
b = a
```

In C, assigning one variable to another makes a copy of the value and put that value in the new box:
![enter image description here](https://lh3.googleusercontent.com/27oN3JtVfNiu4V7QQI58nslrPmHqE6yRVom2NynGkNs=s0 "a2box.png")![enter image description here](https://lh3.googleusercontent.com/p95kf5La-x_HgLqCi5XEE3P3p4hGl4NjjR7rASOFjM0=s0 "b2box.png")

While in Python, this is not the case. Assigning one variable to another, or more precisely in Python, binding a name to another, the new name is just bond to the object bond by the original name, like following
![enter image description here](https://lh3.googleusercontent.com/X4fDmNk_L_Xc-A8OStmEtuC77D_1oNVSXWqglzTEXR8=s0 "ab2tag.png")

We can verify this by the ```id()``` function, which returns the identity[^Note2] of the object:

```python
>>> a = 1
>>> b = a
>>> id(a)
140192199543016
>>> id(b)
140192199543016
```

[^Note2]: An object’s identity never changes once it has been created; you may think of it as the object’s address in memory, please refer to [3] for more details.

So what will happen if we change ```b``` to another value, or more precisely, bind ```b``` to another object?

```python
>>> a = 1
>>> b = a
>>> id(a)
140192199543016
>>> id(b)
140192199543016
>>> b = 2
>>> id(a)
140192199543016
>>> id(b)
140192199542992
```

As we can see, now the name ```b``` is bond to a new object ```2```, while the name ```a``` keep unchanged. Is this always the case? Let's see the following example:

```python
>>> l1 = [1,2,3]
>>> l2 = l1
>>> l2[0] = 4
>>> l1
[4, 2, 3]
>>> l2
[4, 2, 3]
>>> id(l1)
4540233344
>>> id(l2)
4540233344
```

Quite weird, right? After we change the elements in ```l2```, the ```l1``` and ```l2```are still bond to the same object. Why?

In Python, Data Type can be classified into two category: Mutable Type and Immutable Type:

   - Mutable Type: byte array, list, set, dict
   - Immutable Type: int, float, long, complex, str, bytes, tuple, frozen set

The difference can be summarized as following:

```python
x = something # immutable type
print x
func(x)
print x # prints the same thing

x = something # mutable type
print x
func(x)
print x # might print something different

x = something # immutable type
y = x
print x
# some statement that operates on y
print x # prints the same thing

x = something # mutable type
y = x
print x
# some statement that operates on y
print x # might print something different
```

## Shallow Copy v.s. Deep Copy

As we can see in the previous sections, when we copy a variable of mutable type by ```b = a```, then if we change the new variable ```b```, the original variable ```a``` will also be changed. Sometime we want to avoid such side effect, this is how shallow and deep copy can help us:

```python
>>> import copy
>>> l1 = [1, 2, 3]
>>> l2 = copy.copy(l1)
>>> l2
[1, 2, 3]
>>> id(l1)
4540385256
>>> id(l2)
4540386696
>>> l2[0] = 4
>>> l2
[4, 2, 3]
>>> l1
[1, 2, 3]
```

As we can see, by using ```copy.copy()```function, we now bind the name ```b``` to a new object with which contains the same thing while with different identity. However, ```copy.copy()``` may fails when the copied object is a compound object, see below:

```python
>>> import copy
>>> l1 = [1, 2, [3, 4], [5, 6]]
>>> l2 = copy.copy(l1)
>>> l2
[1, 2, [3, 4], [5, 6]]
>>> l2[2] = [8, 9]
>>> l1
[1, 2, [3, 4], [5, 6]]
>>> l2
[1, 2, [8, 9], [5, 6]]
>>> l1 = [1, 2, [3, 4], [5, 6]]
>>> l2 = copy.copy(l1)
>>> l2[2][0] = 10
>>> l1
[1, 2, [10, 4], [5, 6]]
>>> l2
[1, 2, [10, 4], [5, 6]]
```

We can see that if we change the element in the shallow copied object ```l2```, it will not change the corresponding ```l1```. But if we change the element of the element in ```l2```, it will change ```l1```. This is because:
   - A shallow copy constructs a new compound object and then inserts reference into it to the objects found in the original object.
   - A deep copy constructs a new compound object and then, recursively, inserts copies into it of the object found in the original.

```python
>import copy
>l1 = [1, 2, [3, 4], [5, 6]]
>l2 = copy.deepcopy(l1)
>l2[2][0] = 10
> l1
[1, 2, [3, 4], [5, 6]]
> l2
[1, 2, [10, 4], [5, 6]]
```




## Reference

[1]. [Understanding Python variables and Memory Management](http://foobarnbaz.com/2012/07/08/understanding-python-variables/)

[2]. [Drastically Improve Your Python: Understanding Python's Execution Model](https://www.jeffknupp.com/blog/2013/02/14/drastically-improve-your-python-understanding-pythons-execution-model/)

[3]. [Data Model, Python](https://docs.python.org/2/reference/datamodel.html#id5)

[4]. [Data Types, Python Programming](https://en.wikibooks.org/wiki/Python_Programming/Data_Types)

[5]. [Immutable v.s. Mutable Types, Python](http://stackoverflow.com/questions/8056130/immutable-vs-mutable-types-python)


[^Note1]: Everything in Python is a object, even the "integer literal" ```10``` is a object. By run ```dir(10)``` you can get a list of attributes ```10``` has: ```['__abs__', '__add__', '__and__',...,'numerator','real']```.

