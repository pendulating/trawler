"""Cultural name/location perturbation for the PrivacyLens benchmark.

Detects person/location entities in each vignette (a RoBERTa-large NER model),
then applies a deterministic, gender-preserving, record-seeded substitution to swap names and
locations to a target culture. The ``western`` culture is an identity
passthrough (control). The same per-record replacement map is applied across
every text field (S/V seed+vignette JSON, T trajectory dict, sensitive_info_items)
so the vignette stays internally coherent and the leakage judge still works.

Public entrypoint: :func:`perturb_dataset`.
"""

from .perturb_dataset import perturb_dataset

__all__ = ["perturb_dataset"]
