# Logit

The raw unnormalised score a model outputs before a sigmoid or softmax converts it to a probability. Loss functions almost always want the logit rather than the probability, because combining the two steps is numerically stable in a way that computing them separately is not.
