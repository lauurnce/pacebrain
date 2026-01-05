"""API surface checks for pacebrain.inference."""

def test_inference_load_finish_model_load_finish_model_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.inference')
    obj = getattr(mod, 'load_finish_model')
    assert callable(obj)

def test_inference_load_scaler_load_scaler_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.inference')
    obj = getattr(mod, 'load_scaler')
    assert callable(obj)

def test_inference_rebuild_scaler_rebuild_scaler_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.inference')
    obj = getattr(mod, 'rebuild_scaler')
    assert callable(obj)

def test_inference_predict_finish_time_predict_finish_time_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.inference')
    obj = getattr(mod, 'predict_finish_time')
    assert callable(obj)
