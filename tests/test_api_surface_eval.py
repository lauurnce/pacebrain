"""API surface checks for pacebrain.eval."""

def test_eval_mae_minutes_mae_minutes_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.eval')
    obj = getattr(mod, 'mae_minutes')
    assert callable(obj)

def test_eval_rmse_minutes_rmse_minutes_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.eval')
    obj = getattr(mod, 'rmse_minutes')
    assert callable(obj)

def test_eval_riegel_predict_riegel_predict_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.eval')
    obj = getattr(mod, 'riegel_predict')
    assert callable(obj)

def test_eval_get_model_predictions_get_model_predictions_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.eval')
    obj = getattr(mod, 'get_model_predictions')
    assert callable(obj)

def test_eval_plot_comparison_plot_comparison_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.eval')
    obj = getattr(mod, 'plot_comparison')
    assert callable(obj)

def test_eval_run_evaluation_run_evaluation_is_callable():
    import importlib
    mod = importlib.import_module('pacebrain.eval')
    obj = getattr(mod, 'run_evaluation')
    assert callable(obj)
