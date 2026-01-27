"""API surface checks for pacebrain.baselines."""

def test_baselines_mean_prediction_mean_prediction_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.baselines')
    obj = getattr(mod, 'mean_prediction')
    assert callable(obj)

def test_baselines_fit_riegel_scale_fit_riegel_scale_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.baselines')
    obj = getattr(mod, 'fit_riegel_scale')
    assert callable(obj)

def test_baselines_riegel_corrected_prediction_riegel_corrected_prediction_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.baselines')
    obj = getattr(mod, 'riegel_corrected_prediction')
    assert callable(obj)

def test_baselines_fit_linear_fit_linear_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.baselines')
    obj = getattr(mod, 'fit_linear')
    assert callable(obj)

def test_baselines_log_features_log_features_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.baselines')
    obj = getattr(mod, 'log_features')
    assert callable(obj)

def test_baselines_fit_log_linear_fit_log_linear_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.baselines')
    obj = getattr(mod, 'fit_log_linear')
    assert callable(obj)
