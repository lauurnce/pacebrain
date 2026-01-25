# Fine-tuning

Continuing training of a pretrained model on task-specific data, usually at a much lower learning rate. The low rate is the point: a normal rate destroys the pretrained representations in the first few steps, which is the failure mode that looks like the pretrained weights never helped.

## See also

- [Seed](seed.md)
- [Hyperparameter](hyperparameter.md)

In PaceBrain, Fine tuning matters at exactly the boundary between baselines and fancier models — it is one of the knobs that decides whether added complexity earns its keep.

## See also

- [Momentum](momentum.md)
- [Hyperparameter](hyperparameter.md)
