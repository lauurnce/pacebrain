# Teacher forcing

Feeding the true previous token during training rather than the model's own prediction. It speeds convergence and creates exposure bias, because at inference the model must consume its own outputs and has never practised recovering from its own mistakes.
