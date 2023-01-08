
## Learning Objectives

- Use vectorization to implement multiple linear regression
- Use feature scaling, feature engineering and polynomial regression to improve model training
- implement linear regression in code

## Multiple Features

In the original version of linear regression you had a single feature (x).  The model in this case was:

$$f_{w, b}(x) = wx + b$$

For the new notation when there is multiple independt variables that interfere in the prediction we're going to call the:

$$x_1, x_2, x_3 ... x_n$$
 
 To reference a generic feature we can use this notation:

$$x_j = j^{th} feature$$

And the domain that measures the number of features have this generic dimension:

$$j = 1, 2, 3...n$$

The subscript i will reference one specific row of features that was measured for the same sample, that can be called  a row vector. The example below ilustrate that betters:

![[Image 1 - Multiple Feature Example and Notation.png]]

To call out an specific value in a multiple feature problem you will have to give the coordenate of the column (j) and the coornate of the row (i) to accuratly identify the number.

The arrow on top of the feature visually shows that we're talking about a vector.

The generic model for multiple features will be:

$$f_{w, b}(x) = w_1x_1+w_2x_2+w_3x_3...w_nx_n+b$$

The parameters w say the influence of each feature in the model and the parameter b is the base number.

Creating a vector of the parameters w:

$$\vec{w} = [w_1w_2w_3...w_n]$$

Also writing the inputs features as a vector:

$$\vec{x} = [x_{1} x_{2} x_{3} ... x_{n}]$$

The model can be written more succinctly ias:

$$f_{\vec{w}, b} = \vec{w}.\vec{x}+b$$

The dot notation in the model above refers to dot product from linear algebra. The doc products of two vectors of two list of numbers is computed by checking the corresponding pairs of numbers, like that:

$$f_{\vec{w}, b} = \vec{w}.\vec{x}+b = w_1x_1+w_2x_2+w_3x_3...w_nx_n+b$$

As you can see the dot product notation lets you wirte the model in a more compact form, with fewer characters.

The name for this type of linear regression model with multiple input features is multiple linear regression. This is in contrast to univariate regression, which has just one feature. 

This model will be refered as multiple linear regression and not multivariate regression, because that is a name for something else.

## Vectorization

When you're implementing a learning algorithm, using vectorization will both make your code shorter and also make it run much more efficiently. Learning how to write vectorized code will aloow you to also take advantage of modern numerical linear algebra libraries, as well as maybe even GPU () ahrdware that stands for graphics processing unit. This is hardware objectively designed to speed up computer graphics in your computer, but tourns out can be used when you write vectorized code to also help you execute your code much more quickly.

Example of vectorization:

$$\vec{w} = [w_{1} w_{2} w_{3}]$$

$$\vec{x} = [x_{1} x_{2} x_{3}]$$

The number of elements is equal to 3 (n=3). Notice that in linear algebra the index or the counting starts from 1 and so the first value is subscripted 1.
In Python code you can define these variables w, b and x using arrays, like this:

![[Image 2 - Example of how to Declare Vectors Using Python and the Library NumPy.png]]

Above its been used a numerical linear algebra library in Python called NumPy, which is by far the most widely used numerical linear algebra library in Python and in Machine Learning.
Because in Python the indexing of arrays or counting, in arrays, starts from 0, you would access the first value of the parameter using:

$$w_1 = w[0]$$

And the others value follow the same logic:

$$w_2 = w[1]$$ 
$$w_3 = w[2]$$

In a generic way you can say that:

$$w_i = w[i-1]$$

Similarly, to access individual features of the input feature or any vector, for that matter, we use the indexitation above.

Implementation of multiple linear regression without vectorization:

![[Image 3 - Code Example of a Multiple Linear Regression without Vectorization.png]]

You take each parameter w and multiply it by his associated input feature. You could write your code like this, but what if the number of inputs isn't three but instead one hundred or a hundred thousand. The method is inefficient for you to writcode and inefficient for your computer to compute.

Another way of implementing the multiple linear regression without vectorization but using a for loop. 
In math, you can use a summation operator to add all the products of the parameter and his associated input feature, like that:

$$f_{\vec{w}, b}(\vec{x}) = \sum_{j=1}^nw_{j}x_{j}+b$$

The summation goes from j equals to 1 up to n, including n.

In code, using the summation will look like:

![[Image 4 - Example of Multiple Linear Regression Using Summation.png]]

Notice that in Python, the range 0 to n means that j goes from 0 all the way to n-1, and doesn't include n itself. More commonly, this is written range(n) in Python.

While this implementation is a bit better that the first one, still doesn't use vectorization and isn't that efficient.

Now using vectorization:

$$f_{\vec{w}, b} = \vec{w}\vec{x}+b$$

You can implement this with a single line of code:

![[Image 5 - Using Vectorization for Multiple Linear Regression using NumPy.png]]

This NumPy dot function is a vectorized implementation of the dot product operation between two vectors and especially when n is large this will run much faster than the two previous code examples.

Vectorization has two distinct benefits:
First, it makes the code shorter, is now just one line of code.
Second, it also results in your code running much faster, that either of the two previous implementations that did not use vectorization.

The reason that the vectorized implementation is much faster is the behind the scenes, the NumPy dot function is able to use parallel hardware in your computer and this is true wheter you're running this on a normal computer or if you're using a GPU, that's often used to accelerate machine learning jobs.

In a for loop the calculus is being made one step at a time, one step after another. 
In contrast, the dot function from NumPy is implemented in the computer hardware with vectorization. The computer can get all values of the vectors and in a single-step it multiplies each pair with each other in parallel; Then after that, the computer takes these numbers and uses specialized hardware to add them all together very efficiently, rather than needing to carry out distinct additions one after another to add up all numbers in these vectors.

The peformance matters when you're running learning algorithms on large data sets or trying to train large models, which is often the case with machine learning.
That's why being able to vectorize implementations of learning algorithms, has been a key step to getting learning to run efficiently and therefore scale well to alrge datasets that many modern machine learning algorithms now have to operate on.

To apply Gradient Descent without vectorization you will have to calculate the derivative of n values and use the equation of the method n-times for each step. The code for this situation will look like this:

![[Image 6 - Gradient Descent Without Vectorization for Multiple Linear Regression.png]]


In contrast, with vectorization you can imagine the computer's parallel processing hardware like this, it take all the values in vector of the parameter w and subtracts in parallel, alpha times all values in the vector d (vector that contains the derivative values) and return all the calculations back to the w vector, all in the same and all with one step.

Content continues in the note: [[Python, NumPy and Vectorization]]


$$J(\vec{w}, b) = \frac{1}{2m}\sum_{i=1}^m(f_{\vec{w}, b}-\vec{y})^{2}$$

Knowing that the Cost Function is now dependent of multiple features:

$$J(w_{1}, w_{2}, w_{3}, ... w_{n}, b) = J(\vec{w}, b)$$

And the Gradient Descent algorithm has the following format when it's used for multiple linear regression:

$$w_j = w_j - \alpha\frac{\partial}{\partial{w_{j}}}J(w_1, ..., w_n, b)$$

$$b = b - \alpha\frac{\partial}{\partial{b}}J(w_1, w_n, b)$$

The vectorization of The Gradient Descent Algorithm is:

$$w_j = w_j - \alpha\frac{\partial}{\partial{w_{j}}}J(\vec{w}, b)$$

$$b = b - \alpha\frac{\partial}{\partial{b}}J(\vec{w}, b)$$

Changing the derivative for its equation:

$$w_j = w_j - \alpha\frac{1}{m}\sum_{i-1}^m(f_{\vec{w}, b}(\vec{x^{i}})-y^{i})x_{j}^{i}$$

$$b = b - \alpha\frac{1}{m}\sum_{i=1}^m(f_{\vec{w}, b}(\vec{x^{i}})-y^{i})$$

Remebering that the parameters must be simultaneously updated.

There is an alternative way for finding the parametes for linear regression. This metod is call the normal equation. Whereas it turns out Gradient Descent is a great method for minimizing the Cost Function and find its parameters. 
There is one other algorithm that works only for linear regression and pretty much none of the other algorithms you see in this specialization for solving and find the parameters without an interactive gradient descent algorithm.

Some disadvantages of the normal equation method are:
- Unlike Gradient Descent this is not generalized to other learning algorithms, such as the logistic regression algorithm olr the neural network.
- The normal equation method is quite slow if the number of features is large.
- Almost no machine learning practitioners should implement the normal equation method themselves but if you're using a mature machine learning library and call linear regression, there is a chance that in the backend it'll be using this method to solve for the parameters.

The goal of the techinique of **Feature Scaling** enables **Gradient Descent** to run much faster.

Let's start by taking a look at the relationship between the size of a feature, that is how big are the numbers for that feature, and the size of its associated parameter.

When a possible range of values of a feature is large, it's more likely that a good model will learn to choose a relatively small parameter value. Likewise, when the possible values of the feature are small, then a reasonable value for its parameters will relatively large.

To ilustrate that idea well we have the following image:

![[Image 7 - Relation Between Size of the Feature and its Impact in the Parameter.png]]

This is what might end up happening if you were to run **Gradient Descent**, if you were to use your training data as is: because the contours are so tall and skinny the algorithm may end up bouncing back and forth for a long time before it can finally find its way to the global minimum.

In situations like this, a useful thing to do is to scale the features. This means performing some transformation of your training data so the features ranges from 0 to 1, as an example. The key point is that the rescale of the features are both now taking comparable ranges of value to each other.
The **Cost Function** for the dataset rescaled will look like more circles and **Gradient Descent** can find a much more direct path to the global minimum.

One way to apply **Feature Scaling** is to take biggest value from the feature and divide all the values for him:

$$x_{1,scaled} = \frac{x_{1}}{x_{1, max}}$$

In addition to dividing by the maximum, you can also do what's called **Mean Normalization**. 
What this looks like is, you start with the original features and you re-scale them so that both them are centered around zero. So, whereas before they only had values greater than zero, now they have both negative and positive values, that may be usually between negative one and plus one.
So to calculate the mean normalization, first find the average, also called the mean of the number. You can calculate the new features scaled using the mean of the number by the **Mean Normalization** method like that:

$$x_{i, j} = \frac{x_{i, j}-\mu_{j}}{x_{j, max}-x_{j,  min}}$$

Knowing that i range is:

$$i = 1, 2, ..., m$$

You can interprete m as the measurement of the quantity of values that the feature has.

And j is ther number of features that the **Multiple Linear Regression Model** will work, the maximum amount is represent by the letter n.

There is one last commong re-scaling method called **Z-core Normalization**. To implement **Z-Score Normalization** you need to calculate something called the standard deviation of each feature. You will have to first calculate the mean, as well as the standar , which if often denoted by the lowercase Greek alphabet Sigma of each feature.

The equation for the **Z-Score Normalization** method is:

$$x_{i, j} = \frac{x_{i, j}-\mu_{j}}{\sigma_{j}}$$

As a rule of thumb, when performing **Feature Scaling**, you might wanto to aim for getting the features to range from maybe anywhere around negative one to somewhere around plus one, for each feature.
But this range values can be a little bit loose.

You can get a sense of when reascaling is justified to be applied by this image:

![[Image  8 - Scenarios to Evalute Apply Feature Scaling.png]]

When running **Gradient Descent** how can you tell if it's converging? That is, wheter it's helping you to find parameters close to the global minimum of the **Cost Function**. By learning to recognize what a well-running implementation of **Gradient Descent** looks like, we will also be better able to choose a good **Learning Rate**.

As a reminder here's the **Gradient Descent** rule:

$$w_{j} = w_{j}-\alpha\frac{\partial}{\partial{w_{j}}}J(\vec{w}, b)$$

$$b = b-\alpha\frac{\partial}{\partial{b}}J(\vec{w}, b)$$

One of the key choises is the choise of the **Learning Rate**.

Here's something that I often do to make sure that **Gradient Descent** is working well: Recall that the job of the method if find values for the parameters  to hope minimize the **Cost Function**.
What I'll often do is plot the **Cost Function**, which is calculated on the training set, and i plot the value of J at each iteration of **Gradient Descent**. Remember that each iteration means after each simultaneous update, of the parameters. 
**The Cost Function** will be ploted in the vertical axis and the horizontal axis will alocate values of each iteration of the method.
From this plot you may get a curve that looks like this:

![[Image 9 - Example of the Learning Curve.png]]

This curve is also called a **Learning Curve**. Note that there are a few different types of **Learning Curves** used in **Machine Learning**.

If **Gradient Descent** is working properly then the value of the **Cost Function** should decrease after every single iteration. If the value increases after one single iteration, that means either the **Learning Rate** is choosen poorly, and it usually means its value is too large, or there could be a bug in the code.

Another useful thing that this part can tell you in from which iteration the **Cost Function** is levelling off and it's no longer decreasing much. When the levelling of the curve happeng that means that the method is more or less corvengerd.
By the way, the number of iteration that the method can take to converge can vary a lot between different applications.

Another way to decide when your training model is done training is with an **Automatic Convergence Test**. You can define one arbitrary number and if the **Cost Funcation** descreses equals or less between iteration, you can declare that the method has converged. Because you're likely in thos flattened part of the **Learning Curve**.

Your learning algorithm will run much better with an appropriate choice of **Learning Rate**. If's too small it will run very slowly and if it is too large, it may not even converge.

Concretely, if you plot the **Cost Function** for a number of iterations and notice that the Cost sometimes goes up and sometimes goes down, you should take that as a clear sign that **Gradient Descent** is not working properly. This could mean that there's a bug in the code. Or sometimes it could mean that your **Learning Rate** is too large. 
If the **Cost Function** is consistenly increasing after each iteration, this is also likely due to a **Learning Rate** that is too large and it could be addressed by choosing a smaller value. But a **Cost Function** that behavior like this can also be sign of a broken code.

One debugging tip for a correct implementation of **Gradient Descent** is that with a small enough **Learning Rate**, the **Cost Function** should decrease on every single iteration.

So if **Gradient Descent** isn't working, one thing I'll often do is just set the value of the **Learning Rate** to be a very small number and see if that causes the Cost to decrease on every iteration. If even with the **Learning Rate** set to a very small number, the value of the **Cost Function** doesn't decrease on every single iteration, but instead sometimes increases, then that usually means there's a bug somewhere in the code.

A good methodology for choosing the value of the **Learning Rate** is set its initial number as a very small one and gradually increasing its value to see how the it affect the **Cost Function**. You can see an ilustration of a real application of this technique below:

![[Image 10 - Application of Differente Learning Rates.png]]


## Feature Engineering

The choice of features can have a huge impact on your learning algorithm's performance. In fact, for many pratical applications, choosing or entering the right features is a critical step to making the algorithm work well.

Feature engineering: Using intuiton to design new features, by transforming or combining original features.

## Polynomial Regression

This algorithm let's you fit curves, non-linear functions, to your data. 

When you have a dataset that a straight line doesn't appear to fit it very well, you can resort to fit a curve to your data. 

A **Polynomial Regression** is when you take your feature and raised it to a power greather than zero.

When you create features that are these powers, like the square of the original features, then **Features Scaling** becomes increasingly more important.

As feature raised by half or any root operation also be applied to fit the data as a possibility.