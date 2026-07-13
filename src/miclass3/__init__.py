"""Miclass3: reproducible PTB-XL MI proxy classification."""

CLASS_NAMES = ("non_mi", "stemi_proxy", "nstemi_proxy")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

