# Fine-tuning

Continuing training of a pretrained model on task-specific data, usually at a much lower learning rate. The low rate is the point: a normal rate destroys the pretrained representations in the first few steps, which is the failure mode that looks like the pretrained weights never helped.
