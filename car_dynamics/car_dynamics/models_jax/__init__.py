from .dbm import DynamicBicycleModel, DynamicParams, CarState, CarAction
from .models import AdaptDataset, ParamAdaptModel

# The numeric simulator only needs the analytic DBM above.  Load the legacy
# neural backend on demand so its Transformer Engine dependency cannot block
# the Quick Start simulator, while callers that request it still get the real
# import error when its optional dependencies are absent.
def __getattr__(name):
    if name == "DynamicsJax":
        from .nn_dynamics import DynamicsJax

        return DynamicsJax
    raise AttributeError(name)
