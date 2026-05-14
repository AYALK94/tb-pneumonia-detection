"""
TB & Pneumonia Detection Package

This project uses a YOLOv8 model to analyze chest X-ray images
and classify them into one of three categories:

    • Normal
    • Pneumonia
    • Tuberculosis (TB)

Main components:
    - predict.py     → handles model loading and prediction
    - utils.py       → image validation and helper utilities
    - visualize.py   → draws prediction labels on images

The package is designed for use with a FastAPI backend
for medical image inference and testing workflows.
"""

__version__ = "1.0.0"
__author__ = "Ayalkibet Abriham"
__email__ = "ayalkibetabriham94@gmail.com"

from src.predict import load_model, predict_image
from src.utils import (
    validate_image,
    bytes_to_image,
    get_confidence_label,
)
from src.visualize import draw_prediction

# Public exports
__all__ = [
    "load_model",
    "predict_image",
    "validate_image",
    "bytes_to_image",
    "get_confidence_label",
    "draw_prediction",
]