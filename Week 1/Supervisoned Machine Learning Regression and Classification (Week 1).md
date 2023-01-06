
## Learning Objectives

Define machine learning
Define supervised machine learning
Define unsupervised machine learning
Write and run Python code in Jupyer Notebooks
Define a Regression Model
Implement and visualize a cost function
Implement a gradient descent
Optimize a regression model using gradiente descent

## Introduction 

Machine learning can be defined as the science of getting computers to learn without being explicitly programmed to do so.

Machine learning had grown up as a sub-field of AI or artificial intelligence.

"In fact, i find it hard to think of any industry that machine learning is unlikely to touch in a signficant way now or in the near future. Looking even further into the future, many people, including me, are excited about the IA dream of someday building machines as intelligent as you or me. This is sometimes called Aritifical General Intelligence or AGI."

## What is Machine Learning

"Field of study that gives computers the ability to learn without being explicitly programmed"

[[Project Idea I Checkers Machine Learning]]

The two main types of marchine learning are:
Supervised Leaning
Unsupervised Learning

Of this two, supervised learning is the type of machine learning that is used most in many real-word applications and has seen the most rapid advancements and innovation. 

By far, the most used types of learning algorithms today are supervised learning, unsupervised learning and recommender systems.

## Supervised Learning

Machine learning is creating tremendous economic value today. I think 99 percent of the economic value created by machine learning today is through one type of machine learning, which is called supervised learning.

Supervised Machine Learning, or more commonly, Supervised Learning refers to algorithms that learn x to y or input to output mappings.

The key characteristic of supervised learning is that you give your learning algorithms examples to learn from. That includes the "right answers", where by, the right answer means the correct label output for a given input. By seeing correct pairs of input and desired output label that the learning algorithm eventually learns to take the input alone, without the output label, and gives a reasonably accurate prediction or guess of the output.

Examples of supervised machine learning:

![[Image 1 - Examples of Supervised Machine Learning.png]]

![[Image 2 - Correlation Between House Price and Size.png]]

What is more appropiate for to fit the tendence of the data? A curve a straight line or a function even more complex?

What you've seen is an example of supervised learning. Because we gave the algorithm a dataset in which the so called 'right answer' that is the label or the correct price 'y' is given for every house of the plot. The task of the learning algorithm is to produce more of these right answers, specifically predicting what is the likely price of other houses like the 150 square-feet example.

To define a little bit more terminology this house price prediction is the particular type of supervised learning called [regression]. By [regression], meaning that we're trying to predict a number from infinitely many possible numbers.

Take breast cancer detection as an example of a [classification] problem.

![[Image 3 - Example of Supervised Machine Learning of the Classification Type.png]]
![[Image 4 - Further Example of Classification Type.png]]

In opposition of [regression] the number of possibilities in [classification] problems is much small, in this case there are two of them, benign or malignant.

This is differente from [regression] which tries to predict any number, out of the infinitely many number of possible numbers.

In [classification], the terms output classes and output categories are often used as interchageably.

To summarize [classification] algorithms predict categories. Categories don't have to be numbers, they can be non-numeric. For example, we can predict wheter a picture is that of a cat or a dog.

[[Project Idea II identify The animal in the picture]]

But what makes [classification] differente from [regression] when you're interpreting the numbers is that classification predicts a small, finite, limited set of possible output categories such as 0, 1 and 2, but not all possible numbers in between.

You can use more than one input value to predict an output. Like the example below, where te input age in now part of the problem.

![[Image 5 - Example of Classification Problem with Double Inputs.png]]

## Unsupervised Learning

In unsupervised learning the dataset will not have the result label, instead our job is to find some structure, pattern or just something interesting in the data. 

It is called unsupervised because it's not supevised by the algorithm, to generate an answer for every input, instead we ask the algorithm to figure out by himself what it is interesting or what patterns, structures might be in this data.

With the particular dataset below an unsupervised learning algorithm might decide that the data can be assigned to two different groups or two different clusters.

![[Image 6 - Comparios Between a Classification Supervised Learning and Unsupervised Learning.png]]

This is a particular type of unsupervised learning, called a [clustering] algorithm. It is called that way, becasuse it places the unlableds data, into different clusters and this turns out to be used in many applications. For example, [clustering] is used in google news.

![[Image 7 - Application of Clustering Algorithms in Google News.png]]

This example, show that certains article with keywords in their title are clustered in this group of news, without any supervision by the algorithm.

Whereas in supervised learning, the data comes with both inputs x and output label y. In unsupervised learning, the data comes only with inputs x, but not outpus labels y. The Algorithm has to find structure in the data. One example of unsupervised learning is called [clustering], which groups similar data points together.

Other type of unsupervised learning is called [anomaly detection], which is used to detect unusual events. This turns out to be very important for fraud detection in the financial system.

Other form is [dimensionality reduction] that lets you take a big-data set and compress it to a much smaller data-set, while as little information as possible.

## Linear Regression

It's called a supervised learning because you are first training a model by giving a data that has right answers.

The linear [regression] model is a particular type of supervisoned learning model.

It's called [regression] model because it predicts numbers as the output. There are other models for addressing [regression] problems too, the linear is one of them.

In contrast with the [regression] model, the othe most common type of supervised learning model is called [classification] model. The [classification] model predicts categories or discrete categories. In [classification] there are only a small number of possible outputs.

A training set in supervised learning includes both the input features and the output targets. The output targets are the right answers to the model we'll learn from.

To train the model, you feed the training set both the input features and the output targets to your learning algorithm. Then your supervised learning algorithm will produce some function. Historically, this function used to be called hypothesis.

The job of the function is to take a new input and estimate or predict that is nominate ŷ.

The model's prediction is the estimate value of the output target, based in the input features.

In the linear regression model the function adjust to a straight line, with this generic expression:

![[Image 8 -  Generic Equation of the Function that Represents the Linear Regression.png]]

Where: w and b are numbers, their value will determine the prediction of the ŷ based on the input feature x. The w and b can be ommited in the writing of the funcion.

The main advantage is good for foundation and the simplicity. 

The equation above represents a linear regression with one variable or univariable linear regression.

In order, to implement linear regression the first key step is to define something called a [cost function].

The [cost function]  will give the information of how well the model is doing so that we can try to get it and do better.

In the function above the w and b letter represents parameters of the model. In machine learning parameters of the model are the variables you can adjust during training in order to improve the model. Sometimes the parameters are referred to as coefficients or as weights.

In a linear model, this is an ilustration of how the parameters interfere in the behavior of the function:

![[Image 9 - Ilustration of How The Change In Parameters Influence in The Line Behavior.png]]

The line will be accepted as fitting for the training set when you visually can think of this to mean that the line defined by the function is roughly passing through or somewhere close to the training examples as compared to other possible lines that are not as close to these points.

The [cost function] takes the prediction (ŷ) and compares it to the target by taking their subtraction.

$$\sum_{i=1}^m(ŷ^{i}-y^{i})^{2} = error$$

This difference between the prediction and the output target is called the error, in other words, we're measuring how far off the prediction is from the target.

Is computed the square of the difference to avoid negative numbers and the error is meazured in a descrite way, comparing each prediction with each output target.

Finally, we want to measure the error across the entire training set, in particular, let's sum up the squared errors like this.

Remembering that m represents the number of training examples you have at your disposal.

Notice that if we have more training examples m is larger, so the [cost function] will increases in it's value.

To make sure the [cost function] will not get bigger as the samples in the training set increases, by convetion the error is dividide by the number of samples in the dataset. 

$$\frac{1}{m}\sum_{i=1}^m(ŷ^{i}-y^{i})^{2}= error$$

That's called the average squared error.

By convetion, [cost function] that machine learning people use actually divides by two times the number of training samples. The extra division by two is just to make some of the calculations look neater. But the [cost function] still works wheter you include this division by two or not. 

So the final [cost function] is represented this way:

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^m(ŷ^{i}-y^{i})^{2}$$

This function can also be called: Squared error cost function.

In machine learning different people will use different [cost functions] for different applications, but the squared error [cost function] is by far the most commonly used one for linear [regression] and for that matter, for all regression problems, where seems to give good results for many applications. 

We can rearrange the equation above like this:

We know that the prediction value is equal to the result given by the model function, so:

$$f_{w, b}(x^{i}) = wx^{i}+b $$
$$ŷ^{i} = f_{w, b}(x^{i})$$
$$J(w,b) = \frac{1}{2m}\sum_{i=1}^m(ŷ^{i}-y^{i})^{2}$$
$$J(w,b) = \frac{1}{2m}\sum_{i=1}^m(f_{w, b}(x^{i})-y^{i})^{2}$$

Eventually we're going to want to find values of the parameters that make the [cost function] small.

To summarize, to measure how well a choice of the parameters fits the training data you have a cost function (J). That function measures the difference between the model's predictions and the actual true values (y). Linear regression would try to find values for the parameters that make the [cost function] as small as possible.

In math we say that we want to minimize the function.

[Simplified version of the cost function]: Consider the parameter b equals zero, in that cenario the regression line will cross the origen and the model function will only have one parameter to be determine.

The goal of linear regression is to minimize the value of the [cost function] manipulating the numbers of the parameters.

The graphic that evaluate the correlation between the parameter w, when the parameter b in equal zero, and the [cost function] it's shaped like a parable.

When you consider the effect of the parameter b in the discussion  you generate a 3-D Graphic also shaped like a parable, but in the tridimesional plan.

Contour plots are a good way to visualize the impact of the multivariable problem in the [cost function].

In a graphic way the error can me observed as the vertical distance of the dataset dots and the tendency of the linear regression model.

In linear regression rather than having to manually try to read a countour plot for the best value for w and b, which isn't really a good procedure and also won't work once we get to more complex machine learning models. 
What you really want is an efficient algorithm that you can write in code for automatically finding the values of parameters w and b that give you the best fit line. That minimizes the cost function J. 
The algorithm for doing this is called [gradient descent]. This algorithm is one the most important in machine learning.
[Gradient descent] and variations on it are used to train, not just linear [regression], but some of the biggest and most complex models in all of AI.

## Gradient Descent

[Gradient Descent] is used all over the place in machine learning, not just for linear regression, but for training, for example, some of the most advanced neural network models, also called deep leraning models.

[Gradiente Descent] is an algorithm you can use to try to minimize any function, not just a cost function for linear regression. It turns out that this tool applies for more general functions, including other cost functions that work with models that have more than two parameters.

For instance you can have a cost function that have multiple parameters, like:

$$J(w_1, w_2, ..., w_n, b)$$
You can minimize the cost function J that depends of n+1 quantities of parameters.

To start off provide some initial gueses for the parameters w and b. In linear regression in won't matter too much what the initial are, so a commomn choice is to set them both to 0.

The [Gradiente Descent] algorithm will keep on changing the parameter w and b a bit every time to try to reduce the [Cost Function], until hopefully the function sets at or near a minimum.

One important thing to note is that the shape of a [Cost Function] may not be a bow or a hammock, it is possible for there to be more than one possible minimum.

![[Image 10 - Example of a Graph that Represents a Neural Network Model.png]]

The plot generated above is not represent by a [Cost Function] with a squared error format and it's not a linear [Regression].

Mathematically, the [Gradient Descent] method searches for the direction which the steepest descent points to. And the process is repeated as you continue searching throught different points in the [Cost Function].

The initial guess is very important in functions that have more complex formats and more influence of multiple parameters.

The different valleys in the picture above are called local minimum, because depending on the starting point the [Gradient Descent] method will lead you for an especific valley.

[Gradient Descent] Algorithm:

$$w = w-\alpha\frac{d}{dw}J(w, b)$$

What the expression above says is: update your parameter w by taking the current value of w and adjusting it a small amount, which is this expression after the minus sign.

We're assigning w a value using this equal sign. So in this context, this equal sign is the assignment operator. Specifically, if you write code that says a=c, it means take the value c and store it in your computer, in the variable a. So the assignment operator in coding is different than truth assetions in mathematics. 
In Python and in other programming languages, truth assertions are sometimes written as equals equals, meaning that you're testing wheter something is equal other thing. But in math, the equal sign can be used either for assignments or for truth assertion.

The letter alpha from the greek alphabet is called learning rate, in this specific equation. The learning rate is usually a small positive number between 0 and 1. What alpha does is basically control how big of a step you take downhill. 
If alpha is very large then that corresponds to a very aggressive gradient descent procedure.
If alpha is very small then you'd be taking small baby steps downhill.

The term multiplying the learning rate is the derivative termo of the [Cost Function]. You can think that this term dictates what direction you want to take your step.

You also have an assignment operation to update the parameter b:

$$b=b-\alpha\frac{d}{db}J(w, b)$$
For the [Gradient Descent] Algorithm you're going to repeat these two update steps until the algorithm converges. By converges, meaning you reach the point at a local minimum where the parameters w and b no longer change much with each additional step that you take.

You want to simultaneously update w and b, in this method.

How to implement correctly the [Gradien Descent] method:

$$tmp_w = w-\alpha\frac{\partial}{\partial{w}}J(w, b)$$

$$tmp_b = b -\alpha\frac{\partial}{\partial{b}}J(w, b)$$

Calculate each step simultaneously from each parameter.

Then the next w and b will become the temp versions.

Do it until the data converge.

[Simplified version of the gradient descent algorithm]: Consider only the influence of the parameter w.

W way to think of the derivatine in certain point at the line is to draw a tagent line, whcih is a straight line that touches this curve at that point. Enough, the slope of this line is the derivative of the [cost function] at this point.
An to get the slope you can draw a triangle wich connects que tangent line, a parallel line to the x-axis and other parallel line to the y-axis.
When the tangent line is pointing up and to the right, the slope is positive, which means that this derivative is a positive number, so is greater than zero. Considering that the learning rate is always a positive number and if the derivative is positive then the value of the parameter w is decreasing progressively.

In other cenario that the tanget line is pointing down into the right, the slope is negative and the derivative of the [cost function] is negative. And when you update w you get a increase in it's value, because the equation generates a subtraction of a negative product between the learning rate and the derivative.

The choice of the learning rate, will have a huge impact on the efficiency of your implementation of [Gradient Descent].
And if it's value is chosen poorly the method may not even work at all.

For the case where the learning rate is too small, what happens is that you multiply your derivative term by some really, really small number. And so you end up taking a very small baby step in every interection of the method. 
You will need a lot of steps to approach the minimum of the [Cost Function] when the value of the learning rate is too small and the [Gradiente Descent] will work but may be slow in it's performance.

In the situation that the learning rate is too large then you update the parameter to take a very giant step along the [Cost Function] and you may put yourself in a further place from the minimum compared to when you started.
In other words, the [Gradient Descent] algorithm fail to converge or diverge from the solution.

![[Image 11 - Different Cenarios for The Learning Rate and The Impact in the Gradient Descent Algorithm.png]]

When the value of the parameter is already position in the minimum of the function, the tangent line will have a slope that equals zero and as consequence so the term of the derivative, and that will result in a estagnant value for the parameter. Developing in a example:

$$temp_w = w-\alpha\frac{\partial}{\partial{w}}J(W)$$

$$\frac{\partial}{\partial{w}}=0$$

Then:

$$temp_w = w-\alpha\frac{\partial}{\partial{w}}J(W)=w-\alpha.0.J(W)$$

$$temp_w = w$$

The conclusion is that if you are already at a local minimum [Gradient Descent] leaves the parameter unchanged. Because it just updates the new value of the parameter to be the exact same old value.

As we approach the minimum the value of the derivative gets closer to zero, so as we run [Gradient Descent] eventually we're taking very small steps until you finally reach a local minimum.

The derivative of the squared error [Cost Function] with respect to w is this:

$$\frac{\partial}{\partial{w}}J(w, b) = \frac{1}{m}\sum_{i=1}^m(f_{w, b}(x^{i})-y^{i})x^{i}$$

And the derivative with respect to be is:

$$\frac{\partial}{\partial{b}}J(w, b) = \frac{1}{m}\sum_{i=1}^{m}(f_{w, b}(x^{i})-y^{i})$$
The difference betweeen the derivative equations is the x termo multiplying the term in sommatory.

The demonstration of the Calculus will be available in the following archive: [[Cost Function Derivative - Demonstration]].

So here's the [Gradient Descent] algorithm for linear regression:

$$tmp_w = w-\alpha\frac{1}{m}\sum_{i=1}^{m}(f_{w, b}(x^{i})-y^{i})x^{i}$$

$$tmp_b = b -\alpha\frac{1}{m}\sum_{i=1}^m(f_{w, b}(x^{i})-y^{i})$$

You repeatedly carry out these updates to the parameters until convergence.

The [Cost Function] based on the squared error equation only will have a single global minimum point in its plot, so this particular function doesn't have different local minimums. Th technical term for this is that this [Cost Function] is a convex function.

So when you implement [Gradient Descent] on a convex function, one nice property is that so long as your learning rate is chosen properly, it will always converge to the golbal minimum.

The [Gradient Descent] algorithm is classified as Batch when all of the training examples are used in each step of the calculation.
There are other versions of [Gradient Descent] that not look at the entire training set, but instead looks at smaller subsets of the traning data at each update step.