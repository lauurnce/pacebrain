# Parameter counting

Worth doing by hand once per architecture. A linear layer holds `in*out + out`; an LSTM holds `4*(in*hidden + hidden^2 + hidden)`. It makes memory cost predictable and catches architectural mistakes before training does.
