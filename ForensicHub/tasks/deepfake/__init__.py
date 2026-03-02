from .datasets import *
from .models import *
from .transforms import *

_lazy_model_map = {}
_lazy_postfunc_map = {}

# DeepfakeBench is optional - only load if available
DEEPFAKEBENCH_AVAILABLE = False
try:
    from DeepfakeBench.training import detectors
    from DeepfakeBench.training.detectors import __all__ as model_names
    DEEPFAKEBENCH_AVAILABLE = True

    for name in model_names:
        obj = getattr(detectors, name, None)
        if obj is None or not isinstance(obj, type):
            continue

        module_path = getattr(obj, "__module__", None)
        if not module_path or not module_path.startswith("DeepfakeBench.training.detectors"):
            continue

        _lazy_model_map[name] = module_path  # ← 这里直接记录真实 module path，不包 loader
except ImportError:
    pass  # DeepfakeBench not installed, deepfake models won't be available
