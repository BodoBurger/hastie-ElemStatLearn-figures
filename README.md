# Figures from the Elements of Statistical Learning
I try to recreate all figures from the textbook The Elements of Statistical Learning (2nd edition) by Trevor Hastie, Robert Tibshirani, and Jerome Friedman with R using mainly ggplot2 and [mlr](https://github.com/mlr-org/mlr). 

See https://web.stanford.edu/~hastie/ElemStatLearn/ where you can get a PDF of the book.

For some figures you can find the code in the accompanying R package [ElemStatLearn](https://cran.r-project.org/web/packages/ElemStatLearn/index.html), but for many examples/figures in the book it is not immediately clear (at least it is/was for me) how they were produced. I hope it becomes more accessible via the following notebooks:

1. [Introduction](https://github.com/BodoBurger/hastie-ElemStatLearn-figures/blob/master/01-Introduction.md)
2. [Overview of Supervised Learning](https://github.com/BodoBurger/hastie-ElemStatLearn-figures/blob/master/02-Overview-Supervised-Learning.md)
3. [Linear Methods for Regression](https://github.com/BodoBurger/hastie-ElemStatLearn-figures/blob/master/03-Linear-Methods-for-Regression.md)
10. [Boosting and Additive Trees](https://github.com/BodoBurger/hastie-ElemStatLearn-figures/blob/master/10-Boosting-and-Additive-Trees.md)

The notebooks depend on the following **R** packages:

``` r
install.packages(c("ElemStatLearn", "knitr", "ggplot2", "mlr", "directlabels", "ggforce", "gridExtra", "mvtnorm", "reshape2", "scales", "leaps"))
#library("mlr") # machine learning in R
#library("directlabels") # automatic label positioning in ggplot
#library("ggforce") # drawing circles in ggplot
#library("gridExtra") # arrange multiple plots
#library("leaps") # Regression Subset Selection
```

## Links
- general:
    - [mlr Tutorial](https://mlr-org.github.io/mlr/)
    - [How to render R Markdown for github](https://stackoverflow.com/questions/39814916/how-can-i-see-output-of-rmd-in-github)
- graphics and ggplot:
    - [ggplot2 Reference](http://ggplot2.tidyverse.org/reference/)
    - R graph gallery: https://www.r-graph-gallery.com/
    - [Laying out multiple plots on a page](https://cran.r-project.org/web/packages/egg/vignettes/Ecosystem.html)
    - [ggplot2 - Easy Way to Mix Multiple Graphs on The Same Page](http://www.sthda.com/english/articles/24-ggpubr-publication-ready-plots/81-ggplot2-easy-way-to-mix-multiple-graphs-on-the-same-page/)
    - anti-aliasing:
        - https://stackoverflow.com/questions/1898101/qplot-and-anti-aliasing-in-r
        - http://minimaxir.com/2017/08/ggplot2-web/
        - http://gforge.se/2013/02/exporting-nice-plots-in-r/
    - colors:
        - color names in R: http://www.stat.columbia.edu/~tzheng/files/Rcolor.pdf
        - colors in ggplot2: http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
    - annotations and labels:
        - http://directlabels.r-forge.r-project.org
        - https://cran.r-project.org/web/packages/ggrepel/vignettes/ggrepel.html
