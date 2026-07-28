# Why learn this properly

End of year. Setting down what I actually want out of this before starting,
because "learn machine learning" is not a goal, it is a mood.

## The honest starting point

I can call an API and get a prediction. What I cannot do is tell whether the
prediction is any good, or explain why a model behaves the way it does, or
decide whether a problem needs a model at all.

That last one is the gap that bothers me most. The reflex is to reach for a
model because it is the interesting option, not because it is the right one.
Being able to say "a formula does this better" requires understanding both
sides well enough to compare them.

## What I want to be able to do

1. Take a problem, decide whether it is a learning problem, and justify the
   answer.
2. Build a baseline first and be genuinely willing to lose to it.
3. Read a training curve and say what is wrong.
4. Explain why a model is wrong, not just that it is.
5. Ship something small that a real person uses.

Notably absent: reproducing papers, or anything at a scale requiring a GPU
budget. Those are not what I need.

## The approach

Bottom up, and slower than the tutorials suggest.

Statistics and array tooling first — those notes are already written. Then the
learning concepts, then a framework, then one real project carried all the way
to something usable. Not four half-finished projects.

The project matters more than the notes. Concepts read but never applied
evaporate, and I have proven that to myself several times.

## Constraints worth respecting

- **Laptop CPU only.** If it needs a GPU it is out of scope. This is a
  constraint on ambition, and a useful one.
- **A domain I know.** Debugging a model in an unfamiliar field is guesswork —
  you cannot separate a wrong answer from a surprising one.
- **Real users, even if only me.** A model nobody uses never gets the feedback
  that reveals it is wrong.

## Rough sequence for next year

Statistics and tooling (done) → supervised learning concepts → neural network
fundamentals → optimisers and regularisation → sequence models → pick a project
→ build it end to end and write down what happened.

## The thing to avoid

Collecting notes as a substitute for building. Notes are easy and feel like
progress. The point at which this stops being a reading exercise is the point
at which it starts being worth anything.
