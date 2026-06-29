"""HipTR — a TIPSv2 + Qwen VLM for handwritten text recognition.

Kept import-light on purpose: importing ``hiptr`` (and ``hiptr.config`` /
``hiptr.data.alto`` / ``hiptr.data.tokens``) must not require torch, so the ALTO
serialization can be used and tested standalone. Torch-backed modules
(``hiptr.model.*``, ``hiptr.vision.*``, ``hiptr.data.dataset``) are imported
explicitly by the caller.
"""

__version__ = "0.1.0"
