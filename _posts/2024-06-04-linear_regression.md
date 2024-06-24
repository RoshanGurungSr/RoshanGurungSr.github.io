---
title: "Linear Regression with OLS from scratch"
date: 2024-06-04T08:00:30-04:00
categories:
  - Machine Learning
classes: wide
excerpt: Implementation of Linear Regression using OLS technique with mathematical derivation.
use_math: true

---

Linear Regression is a statistical model and a supervised learning algorithm used for predicting a continuous target variable based on one or more predictor variables. The target variable is the final output that we are trying to estimate and the predictor variables are the features of the data. The objective is to find a best-fitting line that minimizes the difference between predicted values, and actual values. 

Simple Linear Regression consists of a single predictor variable “x” and a response variable “y”. It is modeled by the linear equation:

$$ 
y = \beta_0 + \beta_1x + \epsilon \tag{1} \label{eq:simple-lr}
$$

The objective of Simple Linear Regression using the Ordinary Least Square (OLS) method is to find the values of β0 and β1 that minimize the sum of squared differences between the observed value and values predicted. The Sum of Square Error (SSE) is given by:

$$
SSE = \sum_{i=1}^n (y_i - \hat{y_i})^2 \tag{2} \label{eq:sse}
$$

#### 1. Derivation of $\beta_0$ and $\beta_1$
Since our objective is to minimize the SSE, partial derivative w.r.t $\beta_0$ and $\beta_1$ is taken, set them to 0, then we will solve for coeddicients. <br>
Using $\eqref{eq:simple-lr}$ and $\eqref{eq:sse}$, we get:

$$
SSE = \sum_{i=1}^n (y_i - \beta_0 -\beta_1*x_0)^2
$$

Taking partial derivative w.r.t $\beta_0$

$$
\frac{\partial SSE}{ \partial \beta_0} = \frac{\partial \sum_{i=1}^n (y_i - \beta_0 - \beta_1*x_i)^2}{\partial \beta_0}
$$

Using chain rule:

$$
\frac{\partial SSE}{ \partial \beta_0} = 2 * \sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i) * \frac{\partial \sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)}{\partial \beta_0}
$$

$$
= 2 * \sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)
$$

$$
= -2 * \sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i)
$$

Setting partial derivative to 0:

$$
\Rightarrow -2 * \sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i) = 0
$$

$$
\Rightarrow \sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i) = 0
$$

$$
\Rightarrow \sum_{i=1}^ny_i - n\beta_0 - \beta_1\sum_{i=1}^nx_i = 0
$$

$$
\Rightarrow n\beta_0 = \sum_{i=1}^n - \beta_1\sum{i=1}^nx_i
$$

$$
\Rightarrow \beta_0 = \frac{\sum_{i=1}^n y_i - \beta_1\sum_{i=1}^nx_i}{n}
$$

$$
\Rightarrow \beta_0 = \bar{y} - \beta_1\bar{x} \tag{3} \label{eq:beta0} 
$$

Taking partial derivative w.r.t $\beta_1$

$$
\frac{\partial SSE}{\partial \beta_1} = \frac{\partial \sum_{i=1}^n(y_i - \beta_0 - \beta_1x_i)^2}{\partial \beta_1}
$$

Using chain rule:

$$
\frac{\partial SSE}{\partial \beta_1} = -2*\sum_{i=1}^nx_i * (y_i - \beta_0 - \beta_1x_i)
$$

Setting partial derivative to 0:

$$
\Rightarrow -2 * \sum_{i=1}^nx_i * (y_i - \beta_0 - \beta_1x_i) = 0
$$

$$
\Rightarrow \sum_{i=1}^nx_i * (y_i - \beta_0 - \beta_1x_i) = 0
$$

$$
\Rightarrow \sum_{i=1}^nx_iy_i - \beta_0\sum_{i=1}^nx_i - \beta_1\sum_{i=1}^nx_i^2 = 0
$$

Substituting $\beta_0$,

$$
\Rightarrow \sum_{i=1}^nx_iy_i - (\bar{y} - \beta_1\bar{x})\sum_{i=1}^nx_i - \beta_1\sum_{i=1}^nx_i^2 = 0
$$

$$
\Rightarrow \sum_{i=1}^nx_iy_i - \bar{y}\sum_{i=1}^nx_i + \beta_1\bar{x}\sum_{i=1}^nx_i - \beta_1\sum_{i=1}^nx_i^2 = 0
$$

Dividing both sides by n,

$$
\Rightarrow n\sum_{i=1}^nx_iy_i - n\bar{y}\sum_{i=1}^nx_i + n\beta_1\bar{x}\sum_{i=1}^nx_i - n\beta_1\sum_{i=1}^nx_i^2 = 0
$$

We know for the properties of mean,

$$
\sum_{i=1}^nx_i = n\bar{x}
$$

$$
\sum_{i=1}^n = n\bar{y}
$$

Now, using above property,

$$
\Rightarrow \sum_{i=1}^nx_iy_i - \sum_{i=1}^nx_i\bar{y} = \beta_1(\sum_{i=1}^nx_i^2 - \sum_{i=1}^nx_i\bar{x})
$$

$$
\Rightarrow \sum_{i=1}^n(x_iy_i - \bar{y}x_i) = \beta_1\sum_{i=1}^n(x_i^2 - \bar{x}x_i) \tag{4} \label{eq:beta1-dev4}
$$

We know, for covariance of x and y:

$$
Cov(x, y) = \frac{1}{n}*\sum_{i=1}^n(x_i - \bar{x})(y_i - \bar{y})
$$

$$
=\frac{1}{n}\sum_{i=1}^n(x_iy_i - x_i\bar{y} - \bar{x}y_i + \bar{x}\bar{y})
$$

$$
=\frac{1}{n}(\sum_{i=1}^nx_iy_i - \bar{y}\sum_{i=1}^nx_i - \bar{x}\sum_{i=1}^ny_i + \bar{x}\bar{y}\sum_{i=1}^n1)
$$

$$
=\frac{1}{n}(\sum_{i=1}^nx_iy_i - \bar{y}*n\bar{x} - \bar{x}*n\bar{y} + n\bar{x}\bar{y})
$$

$$
=\frac{1}{n}(\sum_{i=1}^nx_iy_i - n\bar{x}\bar{y} - n\bar{x}\bar{y} + n\bar{x}\bar{y})
$$

$$
\Rightarrow \frac{1}{n}(\sum_{i=1}^nx_iy_i - n\bar{x}\bar{y}) = \frac{1}{n}(\sum_{i=1}^nx_iy_i - \bar{x}\bar{y}\sum_{i=1}^n1)
$$

$$
\therefore nCov(x,y) = \sum_{i=1}^n(x_iy_i - \bar{x}\bar{y})
$$

Again,

$$
Var(x) = \frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2
$$

$$
= \frac{1}{n}\sum_{i=1}^n(x_i^2 - 2x_i\bar{x} + \bar{x}^2)
$$

$$
= \frac{1}{n}(\sum_{i=1}^nx_i^2 - 2\bar{x}\sum_{i=1}^nx_i + \bar{x}^2\sum_{i=1}^n1)
$$

$$
= \frac{1}{n}(\sum_{i=1}^nx_i^2 - 2\bar{x}\sum_{i=1}^nx_i + n\bar{x}^2)
$$

$$
= \frac{1}{n}(\sum_{i=1}^nx_i^2 - n\bar{x}^2)
$$

$$
= \frac{1}{n}(\sum_{i=1}^nx_i^2 - x^2\sum_{i=1}^n1)
$$

$$
\therefore nVar(x) = \sum_{i=1}^n(x_i^2 - \bar{x}^2)
$$

We know from equation $\eqref{eq:beta1-dev4}$,

$$
\Rightarrow \sum_{i=1}^n(x_iy_i - \bar{y}n\bar{x}) = \beta_1\sum_{i=1}^n(x_i^2 - \bar{x}*n\bar{x})
$$

$$
\Rightarrow n\sum_{i=1}^n(x_iy_i - \bar{y}\bar{x}) = n\beta_1\sum_{i=1}^n(x_i^2 - \bar{x}^2)
$$

$$
\Rightarrow nCov(x, y) = \beta_1*nVar(x)
$$

$$
\Rightarrow nCov(x, y) = \beta_1*nVar(x)
$$

$$
\therefore \beta_1 = \frac{Cov(x, y)}{Var(x)}
$$

Expanding Cov(x, y) and Var(x),

$$
\beta_1 = \frac{\frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})(y_i - \bar{y})}{\frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2}
$$

$$
\therefore \beta_1 = \frac{\sum_{i=1}^n(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n(x_i - \bar{x})^2}
$$

$$
\therefore \beta_0 = \bar{y} - \beta_1\bar{x}
$$

By solving $\beta_0$ and $\beta_1$ in $\eqref{eq:simple-lr}$, we can calculate the $\hat{y}$. Using $\hat{y}$, we can calculate the SSE, which represents the accuarcy of our simple linear regression.