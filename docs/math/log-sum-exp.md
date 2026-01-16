# Log sum exp

Log-sum-exp computes a maximum smoothly: log(e^a + e^b) = max(a,b) plus a soft correction.

Numerical stability is a modelling concern — overflow silently destroys learning.
