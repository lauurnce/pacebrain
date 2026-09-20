# Docs

Notes that sit alongside the code, grouped by what they are for.

## Planning and background

Written before the modelling, so the decisions are on record before the results could influence them.

- [Problem statement](problem-statement.md): the problem from the PacePack side, before deciding a model was the answer.
- [Learning plan](learning-plan.md): the goal of learning PyTorch by building something usable.
- [Prior art](prior-art.md): a survey of race-time prediction, used to choose the baselines.
- [Data sources](data-sources.md): what running data is obtainable, and what it lacks.
- [Feature design](feature-design.md): the six columns in `FEATURE_COLS` and why each is there.
- [Evaluation plan](evaluation-plan.md): the metric, chosen before seeing which one flatters the result.
- [Risks and anti-goals](risks-and-anti-goals.md): failure modes named up front.
- [Running science](running-science.md): the domain background behind the features.
- [Setup notes](setup-notes.md): environment and dependency decisions.
- [Math refresher](math-refresher.md): the calculus and linear algebra a training loop needs.
- [PyTorch refresher](pytorch-refresher.md): the PyTorch concepts expected to matter, in the order they bite.

## Collections

- [`glossary/`](glossary/): one short entry per concept, with related entries and, where it applies, how the concept shows up in PaceBrain.
- [`math/`](math/): short derivations tied to the code.
- [`deep-dives/`](deep-dives/): walkthroughs of individual functions and classes in `src/pacebrain/`, named `module-name.md`.
- [`reading/`](reading/): notes on papers and books.
- [`journal/`](journal/): dated entries, by month and topic.
