Chapter 1: Introduction
================
Bodo Burger

Examples of learning problems
=============================

Table 1.1. Spam data
--------------------

*skipped for now*

Figure 1.1. Scatterplot matrix of prostate cancer data
------------------------------------------------------

``` r
df.prostate = ElemStatLearn::prostate

pairs(df.prostate[, c("lpsa", "lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45")],
      col = "blueviolet", cex = .5, cex.axis = .5)
```

![](1-Introduction_files/figure-markdown_github/Prostate-1.png)

Figure 1.2. Examples of handwritten digits from U.S. postal envelopes
---------------------------------------------------------------------

![](1-Introduction_files/figure-markdown_github/Zip-plot-1.png)
