from .open import OpenEvaluator
from .qa import QAEvaluator
from .harm import HarmEvaluator
from .ifeval import IFEvaluator
from .mcq import MCQEvaluator
from .ic import ICEvaluator
from .chat import SimplifiedEvaluator
evaluator_mapping = {
    'harm': HarmEvaluator,
    'qa': QAEvaluator,
    'open': OpenEvaluator,
    'ifeval': IFEvaluator,
    'mcq': MCQEvaluator,
    'IC':ICEvaluator,
    "chat":SimplifiedEvaluator
}