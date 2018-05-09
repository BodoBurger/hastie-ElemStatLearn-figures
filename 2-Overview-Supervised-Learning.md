Chapter 2: Overview of Supervised Learning
================
Bodo Burger
2018-05

-   [Linear Model](#linear-model)
    -   [Figure 2.1. Classification with linear regression](#figure-2.1.-classification-with-linear-regression)
-   [Nearest-Neighbor Method](#nearest-neighbor-method)
-   [Sources](#sources)

``` r
knitr::opts_chunk$set(echo = TRUE,
                      message = FALSE,
                      fig.path = "figures/")
set.seed(123)
library("mlr")
library("ggplot2")
```

Linear Model
============

The data is available from the ElemStatLearn package. We start by creating an MLR task and train a linear model to predict y.

``` r
me = ElemStatLearn::mixture.example
df = data.frame(x1 = me$x[,1], x2 = me$x[,2], y = me$y)
knitr::kable(df[sample(200, 4),], row.names = FALSE)
```

|          x1|         x2|    y|
|-----------:|----------:|----:|
|  -0.0771145|  0.5051217|    0|
|   3.1669716|  0.5569466|    1|
|   0.4708088|  0.0254386|    0|
|   1.9454294|  0.4810805|    1|

``` r
tsk = makeRegrTask(data = df, target = "y")
lrn.lm = makeLearner("regr.lm")
mod.lm = train(lrn.lm, tsk)
beta = coefficients(getLearnerModel(mod.lm))
knitr::kable(t(beta))
```

|  (Intercept)|         x1|         x2|
|------------:|----------:|----------:|
|    0.3290614|  -0.022636|  0.2495983|

Figure 2.1. Classification with linear regression
-------------------------------------------------

We first create a grid of points covering all realizations of *x*<sub>1</sub> and *x*<sub>2</sub> in the feature space. These points indicate the prediction of the model in the background of the plot. We set the threshold to .5.

``` r
knitr::kable(summary(df[, c("x1", "x2")])[c(1,6), ], row.names = FALSE)
```

|       x1       |       x2       |
|:--------------:|:--------------:|
| Min. :-2.52082 | Min. :-1.99985 |
| Max. : 4.17075 | Max. : 2.85581 |

``` r
grid = expand.grid(x1 = seq(-2.6, 4.2, .1), x2 = seq(-2.0, 2.9, .05))
y_hat = getPredictionResponse(predict(mod.lm, newdata = grid))
grid["y"] = factor(as.numeric(y_hat > .5))
```

The function for the decision boundary is determined by the estimated coefficients of the model.

``` r
db = function(x1, coef = beta) {
  (coef[3])^-1 * (0.5 - coef[1] - coef[2] * x1)
}
```

Finally, we can plot it!

``` r
ggplot(show.legend = FALSE) + 
  geom_point(aes(x = grid$x1, y = grid$x2, col = grid$y), shape = 20, size = .05, alpha = .5, show.legend = FALSE) +
  geom_line(aes(x = grid$x1, y = db(grid$x1))) +
  geom_point(aes(x = df$x1, y = df$x2, col = factor(df$y)), shape = "o", size = 4, stroke = 2, show.legend = FALSE) +
  scale_colour_manual(values = c("deepskyblue", "orange")) +
  theme_void()
```

![](figures/figure-2.1-1.png)

Nearest-Neighbor Method
=======================

Sources
=======

-   R documentation of mixture.example data `?ElemStatLearn::mixture.example`
-   <https://stackoverflow.com/questions/39822505/drawing-decision-boundaries-in-r>
-   <https://stats.stackexchange.com/questions/21572/how-to-plot-decision-boundary-of-a-k-nearest-neighbor-classifier-from-elements-o>
