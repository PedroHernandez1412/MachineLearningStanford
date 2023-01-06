
## Python and NumPy

Python is the programming language we will be using in this course. It has a set of numeric data types and arithmetic operations. 
NumPy is a library that extends the base capacbilities of Python to add a richer data set including more numeric types, vectors, matrices and many matrix functions. NumPy and Python work together fairly seamlessly. Python aritmetic operators work no NumPy data types and many NumPy functions will accept python data types.

## Vectors

### Abstract

Vectors, as you will use them in this course, are orderered arrays of numbers. In notation, vectors are denoted with lowr case bold letters such as $\mathbf{x}$. The elements of a vector are all the same type. A vector does not, for example, contain both characters and numbers. The number of elements in the array is often referren to as the *dimension* tough mathematicions may prefer *rank*. The vector shown has a dimension of $n$. The elements of a vector can be referenced with an idex. In math setting, indexes typically run from 1 to n. In computer science and these labs, indexing will typically run from to n-1. In notation, elements of a vector, when referenced individually will be indicate the index in a subscript, for example, the $0^{th}$ elemtn, of the vector $\mathbf{x}$ is $x_0$. Note, the x is not bold in this case.


![[Image 100.1 - Vector Notation in Code and Math.png]]

### NumPy Arrays

NumPy's basic data structure is an indexable, n-dimensional *array* contaning elements of the same type (`dtype`). Right away, you may notice we have overloadded the term 'dimension'. Abovei, it was the number of elements in the vector, here, dimension refers to the number of indexes of an array. A one-dimensional or 1-D array has one index. In Course 1, we will represent vectors as NumPy 1-D arrays.

- 1-D array, shape(n,): n elements indexed [0] through [n-1]


### Operations on Vectors

Let's explore some operations using vectors.

#### Indexing

Elements of vectors can be accessed via indexing and slicing. NumPy provides a very complete set of indexing and slicing capabilities. We will explore the basics needed for the course here. 
For more details access [Slicing and Indexing](https://NumPy.org/doc/stable/reference/arrays.indexing.html)
**Indexing** means referring to *an element* of an array by its position within the array.
**Slicing** means getting a *subset* of elements from an array based on their indices.
NumPY starts indexing at zero so the $3^{rd}$ element of an vector  $\mathbf{a}$ is `a[2]`.



