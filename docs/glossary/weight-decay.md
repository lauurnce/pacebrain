# Weight decay

A penalty on parameter magnitude added to the loss, pulling weights toward zero unless the data justifies keeping them large. Equivalent to L2 regularisation for plain SGD but not for Adam, where the two were conflated for years until AdamW separated them.
