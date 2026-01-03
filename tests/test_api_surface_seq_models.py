"""API surface checks for pacebrain.seq_models."""

def test_seq_models_PacingLSTM_forward_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_models')
    obj = getattr(mod, 'PacingLSTM')
    obj = getattr(obj, 'forward')
    assert callable(obj)

def test_seq_models__forward_recurrent__forward_recurrent_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_models')
    obj = getattr(mod, '_forward_recurrent')
    assert callable(obj)

def test_seq_models_PacingGRU_forward_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_models')
    obj = getattr(mod, 'PacingGRU')
    obj = getattr(obj, 'forward')
    assert callable(obj)

def test_seq_models_length_mask_length_mask_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_models')
    obj = getattr(mod, 'length_mask')
    assert callable(obj)

def test_seq_models_masked_mse_loss_masked_mse_loss_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.seq_models')
    obj = getattr(mod, 'masked_mse_loss')
    assert callable(obj)
