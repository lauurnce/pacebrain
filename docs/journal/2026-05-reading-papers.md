# Study notes — getting through a paper faster

Reading linearly is the slowest way to find out whether a paper is relevant.

## The order that works

Title and abstract. Then figures and tables — for empirical work the results
are there and the prose describes them. Then the conclusion. Only then the
method, and only if the result justified it.

## What to look for in the results

What is the baseline, and was it tuned as hard as the proposed method? How many
seeds? Is the improvement larger than the variance? Is the comparison at equal
compute, or does the new method simply use more?

## The ablation table

Usually the most informative table in the paper. It says which parts of the
contribution actually do anything, and it is common for one component to
account for nearly all of the gain.

## The reflex worth building

Ask what would have to be true for this to be wrong, then check whether the
paper addresses it. Most do not, and knowing which question is unanswered is
more useful than remembering the headline number.
