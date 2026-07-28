# Study notes — plotting for diagnosis

Companion to the numpy/pandas notes. Plots here are for working out what is
going on, not for presentation.

## The two figures worth knowing

**Line plot over time** — for anything that should trend. A training curve is
this, and reading one is a skill in itself: two series on the same axes, and
the *gap* between them is usually more informative than either line.

**Scatter of predicted vs actual** — the single most useful diagnostic for a
regression. Plot the identity line and the structure of the errors becomes
visible immediately:

- Points hugging the line: good.
- A consistent offset above or below: systematic bias, which an averaged error
  metric will report as "some error" without revealing that it is all in one
  direction.
- A fan widening to the right: error proportional to magnitude rather than
  constant.
- Curvature: the model is missing a nonlinearity.

None of that is visible in a single summary number, which is the argument for
always plotting before trusting a metric.

## Backends

Matplotlib chooses an output backend at import. On a machine with no display,
or in a script that only saves files, the interactive default can fail or hang.

Setting `matplotlib.use("Agg")` selects a non-interactive backend that renders
straight to file. It must happen **before** `pyplot` is imported, since the
backend is bound at that point. That single ordering constraint is worth
writing down because it makes the import order in a plotting script look wrong
to a linter, and correcting it breaks the script.

## Habits worth keeping

- **Label axes and include units.** A plot revisited a week later without them
  is close to useless.
- **Equal aspect ratio for predicted-vs-actual.** Otherwise the identity line
  is not at 45 degrees and the visual intuition breaks.
- **Save rather than show** when scripting. A file can be compared against last
  week's; a window cannot.
- **Put the metric in the title.** The number and the picture belong together —
  reading either alone loses information the other has.

## The point

Summary statistics compress, and compression discards exactly the structure
that explains a failure. A metric says how wrong; a plot usually says why.
