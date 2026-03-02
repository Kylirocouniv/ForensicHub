def get_all_lazy_maps():
    # 在函数内部导入，避免循环
    from ForensicHub.tasks.aigc.models import _lazy_model_map as _aigc_model_map
    from ForensicHub.tasks.aigc.models import _lazy_postfunc_map as _aigc_postfunc_map
    from ForensicHub.tasks.document.models import _lazy_model_map as _document_model_map
    from ForensicHub.tasks.document.models import _lazy_postfunc_map as _document_postfunc_map
    from ForensicHub.tasks.deepfake import _lazy_model_map as _deepfake_model_map
    from ForensicHub.tasks.deepfake import _lazy_postfunc_map as _deepfake_postfunc_map

    model_map = {}
    model_map.update(_aigc_model_map)
    model_map.update(_document_model_map)
    model_map.update(_deepfake_model_map)

    postfunc_map = {}
    postfunc_map.update(_aigc_postfunc_map)
    postfunc_map.update(_document_postfunc_map)
    postfunc_map.update(_deepfake_postfunc_map)

    return model_map, postfunc_map


def is_deepfakebench_available():
    """Check if DeepfakeBench is installed."""
    try:
        from ForensicHub.tasks.deepfake import DEEPFAKEBENCH_AVAILABLE
        return DEEPFAKEBENCH_AVAILABLE
    except ImportError:
        return False


def _wrap_deepfake_if_needed(cls, name):
    """如果是 deepfake 模型，则包一层 Wrapper"""
    from ForensicHub.tasks.deepfake import _lazy_model_map as _deepfake_model_map
    from ForensicHub.tasks.deepfake import _lazy_postfunc_map as _deepfake_postfunc_map

    flag = False

    if name in _deepfake_model_map:
        # Check if DeepfakeBench is available
        if not is_deepfakebench_available():
            raise ImportError(
                f"Model '{name}' requires DeepfakeBench which is not installed. "
                "Please install DeepfakeBench to use this model."
            )

        from ForensicHub.tasks.deepfake.wrapper.wrappers import Deepfake2ForensicWrapper
        from DeepfakeBench.training import detectors

        flag = True
        cls = getattr(detectors, name)

        class WrappedModel(Deepfake2ForensicWrapper):
            def __init__(self, *args, **kwargs):
                super().__init__(cls, *args, **kwargs)

        WrappedModel.__name__ = f"Wrapped{name}"
        WrappedModel.__qualname__ = WrappedModel.__name__
        return WrappedModel, flag

    return cls, flag
