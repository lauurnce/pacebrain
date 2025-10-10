# Study notes — numpy and pandas

The tooling layer. Nothing here is conceptually hard, but not knowing it makes
everything else slow.

## Why arrays instead of lists

A Python list of numbers stores boxed objects and pointers. A numpy array
stores a contiguous block of one dtype. That means operations run in compiled
code over the whole array rather than a Python loop per element — typically one
to two orders of magnitude faster.

The habit to build: **express operations over whole arrays, not element by
element.** A visible `for` loop over rows is usually a sign of doing it wrong.

## Shape

Shape is the tuple of dimensions, and most errors are shape errors.

Two that cause real confusion:

- `(n,)` is a 1-D array. `(n, 1)` is a 2-D column. They are not the same, and
  mixing them silently triggers broadcasting instead of erroring.
- `(1, n)` is a row, not a column.

`reshape(-1, 1)` turns a flat array into a column, where `-1` means "infer this
dimension".

## Broadcasting

Operations between different shapes stretch the smaller one, comparing
dimensions right to left; each must match or be 1.

Useful — subtracting a length-`k` mean from an `(n, k)` matrix just works. Also
dangerous: subtracting `(n,)` from `(n, 1)` produces an `(n, n)` matrix rather
than an error. Silently, and with plausible-looking contents.

## dtype

`float64` is numpy's default. Many other tools default to `float32`. Mixing
them either errors at a boundary or silently upcasts and doubles memory. Decide
once, cast at the edge.

## pandas

A DataFrame is labelled columns of arrays, which makes it the right format for
reading messy files and the wrong format for arithmetic in a loop.

Points worth remembering:

- **`.values` / `.to_numpy()`** drops out to the array layer for computation.
- **The index is not a column.** Filtering keeps original index values, so
  positional assumptions break afterwards. `reset_index(drop=True)` after any
  filter or reorder saves confusion later.
- **Chained indexing** (`df[a][b] = x`) may write to a temporary copy rather
  than the frame. Use `.loc`.
- **Missing data** propagates: `NaN` in an arithmetic chain contaminates the
  result rather than raising. Check for it explicitly at load time rather than
  discovering it downstream.

## Reading real files

Real CSVs have missing values, inconsistent types, and columns named
differently than expected. `pd.read_csv` will happily load nonsense — it infers
a dtype per column, so one stray non-numeric value turns a whole numeric column
into strings.

The lesson: validate what you loaded before using it. Check columns exist,
check dtypes, check row counts. Failing loudly at load is much cheaper than
debugging a wrong number three steps later.
