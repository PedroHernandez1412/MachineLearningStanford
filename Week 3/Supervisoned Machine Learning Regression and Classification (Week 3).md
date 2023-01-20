# Classification

In **Classification** your output can take on only a small handful of possible values instead of any number in a infinite range of numbers.

It turns out that **Linear Regression** is not a good algorithm for **Classification** problems.

Below are some examples of **Classification** problems:

![[Image 01 - Example of Binary Classification Problems.png]]

In each of these problemas the variable that you want to predict can only be one of two possible values. This type of **Classification** problem where there are only two possible output is called **Binary Classification**.
Where the word binary refers to there being only two possible classes or two possible categories.

In these problemas the terms class and category are relatively interchangeably.

We often designate classes as "yes" or "no" or sometimes equivalently "true" or "false", or very commonly using the numbers "one/1" or "zero/0".
Following the common convention in computer science with zero denoting false and one denoting true.

One of the terminologies commonly used is to call the false or zero class, the negative class and the true or one class, the positive class.

Negative and positive do not necessarily mean bad versus good or evil versus good. It's just that negative and positive examples are used to convey the concepts of absence or zero false vs the presence or true or one of something you might be looking for.

Wheter certain output is considered the positive or negative class is often arbitrary.

**Linear Regression** is not applied for this type of problem because will not fit well a binary set of outputs, so the main method used for this application is called **Logistic Regression**.

One thing confusing about the name **Logistic Regression** is that even though it has the word regression in it it's actually used for **Classification**.
The name was given for historical reasons. But the method it's actually used to solve **Binary Classification** problems when the output label is either zero or one.

## Logistic Regression

**Logistic Regression** is probably the single most widely used **Classification** algorithm in the world. 

In contrast of **Linear Regression**, what **Logistic Regression** will end up doing is fit a curve that has a "S"  shape, like shown in the example below, to the dataset.

![[Image 02 - Shape of the Logistic Regression Curve.png]]

The output label wil never be an intermediate number between 0 and 1, in the example given. Its value only ever be 0 or 1.

The **Sigmoid Function** of the **Logistic Function** has the appropriate shape to predict the behavior of a **Classification** problem.

The **Sigmoid Function** outputs values in between 0 and 1. And you can use g(z) to denote this function, then you have the generic formula for this curve, like this:

$$g(z) = \frac{1}{1+e^{-z}}$$

Where here e is a matematical constant that takes on a value about 2.7.

The value of the **Sigmoid Function** ranges from zero to 1.

When the value of z is large, the functiob value is going to be very close to zero, that is effect of the constant be elevated by a very large number that have the negative sign in front of it, and that's equivalent of the fraction between 1 and z.
When z is a very large negative number, then g becomes a fraction between 1 over a giant number, which is why g is very close to zero, in that situation.

That's why the **Sigmoid Function** has this unique shape, where it starts very close to zero and slowly builds up or grows to the value of one.

Also in the **Sigmoid Function** when z is equal to 0, then g is equal to 0.5.


To build up the **Logistic Regression** algorithm we will use two steps:
- In the first step, we will store the value of the **Multiple Linear Regression** model in a variable called z:

$$z = \vec{w}.\vec{x} + b$$

- The next step then is to take this value of z and pass it to the **Sigmoid Function**:

$$g(z) = \frac{1}{1+e^{-z}}$$

When you take these two equations and put them together, tehy then give you the **Logist Regression Model**:

$$f_{\vec{w}, b}(\vec{x}) = g(\vec{w}\vec{x} + b)$$

What the **Logistic Regression Model** does is it inputs a feature or a set of features and outputs a number between zero and one.

A interpretation of the **Logistic Regression Model** output is to think of it as the probability that the class or the label output be equal to 1 given a certain input.

If the output of a model is 0.7, that means that the model is predicting orr the model thinks there's a 70% chance that the true label would be equal to 1 for this patient.
To compute the chance of the output being zero, in the example above, we do this calculation:

$$P(y = 0) + P(y = 1) = 1$$

$$P(y = 0) + 0.7 = 1$$

$$P(y = 0) = 1-0.7 = 0.3$$

In other words, there's a 30% chance of the output label be equal zero.

This notation can be used to represent **Logistic Regression**:

$$f_{\vec{w}, b}(\vec{x}) = P(y = 1 | \vec{x}; vec{w}, b)$$

You can read the notation above as the probability of y is 1, ggiven input $\vec{x}$, parameters $\vec{w}$ and $b$.

## Decision Boundary

Now, what if you want the learning algorithm to predict if the value of y is going to be zero or one? 
One thing you might do is set a threshold above which you predict y is one or you set ŷ to be equal to one and below which my prediction is going to be qual zero.

A common choice would be to pick a threshold of 0.5.

If  $f_{\vec{w}, b}{\vec{x}} ≥ 0.5?$ Yes, then: $ŷ = 1$
If $f_{\vec{w}, b}(\vec{x})≥ 0.5?$ No, then: $ŷ = 0$

Whenever zero is equal or greater than zero the **Logistic Regression Model** will result in a value equal or greater than 0.5, so:

If  $f_{\vec{w}, b}{\vec{x}} ≥ 0.5?$ When: $z ≥ 0$
If $f_{\vec{w}, b}(\vec{x})≥ 0.5?$ When: $z<0$

The **Decision Boundary** is the line that mathematically is represented by this result:

$$z = \vec{w}.\vec{x}+b = 0$$

That's the line where you're just almost neutral whether $y$ is zero or one.

The parameters defines where the **Decision Boundary** will be placed in the plot.

Inside the shape of the **Decision Boundary** the prediction output will be defined as 1 and outside its limitation the preditict it will be equal zero.

## Cost Function

The **Cost Function** gives you a way to measure how well a specific set of parameters fits the training data. Thereby gives a way to try to choose better parameters.

As before we'll use $m$ to denote the number of training samples.

The sample size will range in this way:

$$i = 1, 2, 3, ..., m$$

Each training examples has one or more features, for a total of $n$ features.

The index of the number of features can be represented like that:

$$j = 1, 2, 3,...,n$$

And in a **Binary Classification** task the target label $y$ takes on only two values, either 0 or 1.

The **Logistic Regressions Model** is defined by this equation:

$$f_{\vec{w}, b}({\vec{x}}) = \frac{1}{1+e^{-\vec{w}\vec{x} + b}}$$

When you try to plot the **Squared Error Cost Function** using the  **Logistic Regression Model** the result will be a non-convex function, like that:

![[Image 03 - How the Plot of the Squared Error Cost Function Would Look Like Using Logistic Regression.png]]


If you were to use **Gradient Descent** there are lots of local minimum that you can get sucked in.

For application in **Logistic Regression** the **Squared Error Cost Function** will be modified.

At first, we will call this term inside the summation the loss on a single training example:

$$L(f_{\vec{w}, b}({\vec{x}^{(i)}}), y^{(i)}) = \frac{1}{2}(f_{\vec{w}, b}(\vec{x^{(i)}})-y^{(i)})^{2}$$

You can see that the **Loss** is defined with this capital L and is a function of the prediction $f_{\vec{w}, b}(\vec{x^{(i)}})$ of the learning algorithm as well as of the true label $y$.

By choosing a different form for this **Loss Function**, we will be able to keep the overall **Cost Function**, to be a convex function.

The definition of the **Logistic Cost Function** is:

When $y^{(i)}  = 1$:

$$L(f_{\vec{w}, b}(\vec{x^{(i)}}), y^{(i)}) = -log(f_{\vec{w}, b}(\vec{x^{(i)}}))$$

But when $y^{(i)} = 0$:

$$L(f_{\vec{w}, b}(\vec{x^{(i)}}) = -log(1-f_{\vec{w}, b}(\vec{x^{(i)}}))$$

Remember that the **Loss Function** measures how well you're doing on one training example and is by summing up the losses on all of the training examples that you then get the **Cost Function**, which measures how well you're doing on the entire training set.

The plotting of the **Lost Function** in this differente values of labels can be visualized below:

![[Image 04 - Loss Function when the Label Output is Equal to 1.png]]



If the algorithm predicts a probability close to 1 and the true label is equal 1, then the loss is very small, it's pretty much zero.

In the cenario that the output label is equal to 1, we have this two distincts situations:

As $f_{\vec{w}, b}(\vec{x^{(i)}})$ -> $1$ the loss -> $0$

As $f_{\vec{w}, b}(\vec{x^{(i)}})$ -> $0$ the loss -> $∞$

In the cenario that the output label is equal to 0, we have this two distincts situations:

As $f_{\vec{w}, b}(\vec{x^{(i)}})$ -> $1$ the loss -> $∞$

As $f_{\vec{w}, b}(\vec{x^{(i)}})$ -> $0$ the loss -> $0$

Emphasizing, that the value of the prediction will always have values between 0 and 1, in this specific **Binary Classification Problem**.

*Mathematical notes*: The argument of an logarithm operation cannot be negative thereby the value of the **Logistic Regression Model** will never be higher than 1, in the situation that the outpout label is equal zero. Remember that logarithm operations can represent that:

$$log(X) = A$$

Is the same as:

$$10^{A} = X$$

The further the prediciton $f_{\vec{w}, b}(\vec{x^{(i)}})$ is from the target $y^{(i)}$, the higher the loss.

It turns out that with this choice of **Loss Function**, the overall **Cost Function** will be convex and thus you can reliably use **Gradient Descent** to take you to the global minimum.

You may remember that the **Cost Function** is a function of the entire training set and it is therefore the average of the **Loss Function** on the individual training examples.

If you can find a value of the parameters $w$ and $b$ that minimize this, then you'd have a pretty good set of values for the parameters for **Logistic Regression**.

To concantenate the two differents cenarios of the **Loss Function** in **Binary Classification** problems we can write its equation in this way:

$$L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)}) = -y^{(i)}log(f_{\vec{w}, b}{\vec{x^{(i)}}})-(1-y^{(i)})log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})$$

Knowing that the output label can only takes values that equal either 1 or 0. 
The first case will result in this equation:

$$L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)}) = -y^{(i)}log(f_{\vec{w}, b}{\vec{x^{(i)}}})-(1-y^{(i)})log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})$$

$$y^{(i)} = 0$$

Then:

$$L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)}=0) = -0log(f_{\vec{w}, b}{\vec{x^{(i)}}})-(1-0)log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})$$

$$L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)}=0) = 0-1log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})$$

$$L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)}=0) = -log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})$$

The second case will result:

$$L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)}) = -y^{(i)}log(f_{\vec{w}, b}{\vec{x^{(i)}}})-(1-y^{(i)})log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})$$

$$y^{(i)} = 1$$

Then:


$$L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)}=1) = -1log(f_{\vec{w}, b}{\vec{x^{(i)}}})-(1-1)log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})$$

$$L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)}=1) = -1log(f_{\vec{w}, b}{\vec{x^{(i)}}})-0log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})$$

$$L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)}=1) = -log(f_{\vec{w}, b}{\vec{x^{(i)}}})$$

The **Cost Function** writen for **Logistic Regression** will look like this:

$$J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^{m}[L(f_{\vec{w}, b}{\vec{x^{(i)}}}, y^{(i)})]$$

$$J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^{m}[-y^{(i)}log(f_{\vec{w}, b}{\vec{x^{(i)}}})-(1-y^{(i)})log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})]$$

$$J(\vec{w}, b) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(f_{\vec{w}, b}{\vec{x^{(i)}}})+(1-y^{(i)})log(1-f_{\vec{w}, b}{\vec{x^{(i)}}})]$$

This particular **Cost Function** is derived from statistics using a statistical principle called maximum likelihood estimation, which is an idead from statistics on how to efficiently find parameters for different models.

## Gradient Descent

To fit the parameters of a **Logistic Regression Model** we're going to try to find the values of the parameters $w$ and $b$ that minimizes the **Cost Function** $J(\vec{w}, b)$ and we'll again apply **Gradient Descent** to do this.

Here again is the **Cost Function** for **Classification** problems:

$$J(\vec{w}, b) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(f_{\vec{w,b}}(\vec{x^{(i)}})) + (1-y^{(i)})log(1-f_{\vec{w,b}}(\vec{x^{(i)}}))]$$

So if you want to minimize the cost $J$ as a function of $w$ and $b$, here's the usual **Gradient Descent** algorithm:

$$w_j = w_j - \alpha\frac{\partial}{\partial{w_j}}J(\vec{w}, b)$$

$$b = b - \alpha\frac{\partial}{\partial{b}}J(\vec{w},b)$$

Remember that this parameters will be repeatedly updated each parameter as his own value minus the learning rate ($\alpha$) times this derivative term.

To see the derivative demonstration click here: [[Derivative of the Logisitic Cost Function]]

Similarly as **Linear Regression** the derivative will equations will have the following format:

$$\frac{\partial}{\partial{w_j}}J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w,b}}(\vec{x^{(i)}}-y^{(i)})x^{(i)}$$

$$\frac{\partial}{\partial{b}}J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w,b}}(\vec{x^{(i)}}-y^{(i)})$$

The way to carry out these updates is to use simultaneous updates, meaning that you first compute the right-hand side for all of these updates and then simultaneously overwrite all the values on the left at the same time.

Plugging the derivative expressions in the **Gradient Descent** algorithm:

$$w_j = w_j - \alpha\frac{\partial}{\partial{w_j}}J(\vec{w}, b)$$

$$w_j = w_j - \alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w,b}}(\vec{x^{(i)}}-y^{(i)})x^{(i)}$$

$$b = b - \alpha\frac{\partial}{\partial{b}}J(\vec{w},b)$$

$$b = b - \alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w,b}}(\vec{x^{(i)}}-y^{(i)})$$

The expression for the derivative in **Linear Regression** and **Logistic Regression** are the sime but the way to compute the model function in different.

In **Linear Regression**:

$$f_{\vec{w}, b}(\vec{x^{(i)}}) = \vec{w}.\vec{x} + b$$

In **Logistic Regressions**:

$$f_{\vec{w}, b}(\vec{x^{(i)}}) = \frac{1}{1+e^{-(\vec{w}.\vec{x}+b)}}$$

All the same concepts from **Linear Regression** applis for **Logistic Regression**:

- Monitor **Gradient Descent** using the **Learning Curve**;
- Vectorized Implementation
- Feature Scaling

# Overfitting

When the model does not fit the training data very well the technical term for this is the model is **Underfitting** the training data.

An example of **Underfitting** can be visualized in the image below:

![[Image 05 - Example of Underfitting.png]]

Another term for it is the algorithm has high bias.

In machine learning, the term bias has multiple meanings. In this case the term bias refers to when the algorithm has underfit the data, meaning that it's just not even able to fit the training data set that well. There's a clear pattern in the training data that the algorithm is just enable to capture.

Another way to think of this form of bias is as if the learning algorithm has a very strong peconception, or we say a very strong bias, that the housing prices are going to be a completely linear function of the size despite data to the contrary. This preconception that the data is linear causes it to fit a straight line that fits the data poorly, leading it to underfitting data.

When you want examples that are outside of the training set to have an appropiate fit, we call that situation **Generalization**.
Technically we say that you want your learning algorithm to generalize well, which means to make good predictions even on brand new examples that it has never seen before.

When the model fits the traning set extremely well, sometimes it fits the data almost too well, hence is overfit. That results that the model will note generalize to new examples that's never seen before:

An example of **Overfitting** can be seem in the image right below:

![[Image 06 - Example of Overfitting.png]]

Another term for **Overfitting** is that the algorithm has high variance. In machine learning many people will use the terms **Overfit** and high-variance almost interchangeably. We'll use the terms **Underfit** and high bias almost interchangeably.

The intuiton behind **Overfitting** or high-variance is that the algorithm is trying very very hard to fit every single training example. And it turns out that if your training set were just even a little bit different, then the function that the algorithm fits could end up totally different.

The goal of machine learning is to find a model that is hopefully neither underfitting nor overfitting. In other words, a model that has neither high bias nor high variance.

So far we've look **Underfitting** and **Overfitting** for **Linear Regression** algorithms.

Simlarly, **Overfitting** applies to **Classification** as well. The image next show the examples in a **Classification** context:

![[Image 07 - Underfitting and Overfitting in a Classification Context.png]]


## Adressing Overfitting##

One way to address an **Overfitting** problem is to collect more training data, that's one option. If you are able to get more data, then with the larger training set, the learning algorithm will learn to fit a function that is less wiggly. 
To summarize, the number one tool you can use against **Overfitting** is to get more training data.

A second option for addressing **Overfitting** is to see if you can use fewer features. it turns out that if you have a lot of features, but you don't have enough training data, then the learning algorithm may also **Overfit** to your training data.
Choosing the most appropriate set of features to use is sometimes also called *feature selection*. One way you can do so is to use your intuition to choose what you think is the best set of features, what's the most relevant for the prediction.
One disadvantage of *feature selection* is that by using only a subset of the features, the algorithm is throwing away some of the information that you habve about the output.

The third option to reduce **Overfitting** is called **Regularization**. 

**Regularization** is a way to more gently reduce the impacts of some of the features without doing something as harsh as eliminating it outright.
What **Regularization** does is to encourage the learning algorithm to shrink the values of the parameters without necessarily demanding that the parameter is set ot exactly 0. So the technique let's you keep all of the features, but it just prevents the feature from having an overly large effect, which is sometimes what causes **Overfitting**.

By convention we often reduces just the size of the parameters $w_j$. It doesn't make a huge diference wheter you regularize the parameter $b$ or not.

## Regularization

**Regularization** tries to make the parameters values $w_1$ through $w_n$ small to reduce **Overfitting**.

The idea behind regularization is that if there are smaller values for the parameters, then that's a bit like having a simpler model. Maybe one with fewer features, which is therefore less prone to **Overfitting**.

More generally, the way that **Regularization** tends to be implemented is if you have a lot of features, you may not know which are the most important features and which ones to penalize. So the way **Regularization** is typically implemented is to penalize all of the features or more precisely, you penalize all the $w_j$ parameters and it's poissible to show that this will usually result in fitting a smoother, simpler, less weekly function that's less prone to **Overfitting**.

Because we don't know which of these parameters are going to be the important ones. Let's penalize all of them a bit and shrink all of them by adding this newr term below:

$$J(\vec{w}, b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}) - y^{(i)})^{2} + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$$

$$\lambda > 0$$

The value $\lambda$ here is the Greek alphabet lambda and it's also called a *Regularization Parameter*. Similar to picking a *Learning Rate* rate $\alpha$ you now also have to choose a number for $\lambda$.

By convention we also divide the $2m$ , so both the summation terms are scaled in the same proportion. It turns out that by scaling both terms the same way it becomes a little bit easier to choose a good value for $\lambda$.

And in particular you find that even if your training set size grows, the same size of $\lambda$ that you've picked previously is now also more likely to continue to work if you have this extra sling by $2m$.

And by the way, bi convention we're not going to penalize the paramter b for being large. In practice, it makes very little difference wheter you do or not.

If $\lambda = 0$ the model will likely **Overfit** and if the value of $\lambda$ is enormous, the model will **Underfit**.
And so what you want is some value of $\lambda$ that is in between that more appropriately balance these first and second terms of  trading off, minimizing the mean squared error and keeping the parameters small.

To apply **Gradient Descent** with **Regularization** in **Linear Regression** the expression of the derivatives changes a bit:

$$J(\vec{w}, b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}) - y^{(i)})^{2} + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$$

$$w_j = w_j - \alpha\frac{\partial}{\partial{w_j}}J(\vec{w}, b)$$

$$\frac{\partial}{\partial{w_j}}J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}-y^{(i)})x_{j}^{(i)}+\frac{\lambda}{m}w_j$$

$$b = b - \alpha\frac{\partial}{\partial{b}}J(\vec{w},b)$$

$$\frac{\partial}{\partial{w_j}}J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}-y^{(i)})$$

Putting the regularized definitions of the derivatives and putting them back in the **Gradient Descent** expressions for updating the parameters:

$$w_j = w_j - \alpha[\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}-y^{(i)})x_{j}^{(i)}+\frac{\lambda}{m}w_j]$$

$$b = b - \alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}-y^{(i)})$$

As usual, please remeber to carry out simultaneous updates for all of these parameters.

We can manipulate the update term of the parameter $w$ like that:

$$w_j = w_j-\alpha\frac{\lambda}{m}w_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}-y^{(i)})x_{j}^{(i)}$$

$$w_j = w_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}-y^{(i)})x_{j}^{(i)}$$

The second term is exactly the same as the usual **Gradient Descent** update for unregularized **Linear Regression**.

The effect of the therm multiplying the old value of the parameter $w$ is that on every single iteration of **Gradient Descent** you're taking $w_j$ and multiplying it by number slightly lower than 1, before carrying out the usual update.

The update for the *regularized* **Logistic Regression** is very similar of the **Linear Regression**.

The *regularized*  **Cost Function** for **Logistic Regression** can be expressioned this way:

$$J(\vec{w}, b) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(f_{\vec{w,b}}(\vec{x^{(i)}})) + (1-y^{(i)})log(1-f_{\vec{w,b}}(\vec{x^{(i)}}))] + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$$

$$w_j = w_j - \alpha[\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}-y^{(i)})x_{j}^{(i)}+\frac{\lambda}{m}w_j]$$

$$b = b - \alpha\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x^{(i)}}-y^{(i)})$$

The parameter $b$ also will not be updated using **Regularization** in **Logistic Regression**.

The main difference between is that the model for the function in **Logictic Regressions** is the **Sigmoid Function**.