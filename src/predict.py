from ultralytics import YOLO
from PIL import Image
import numpy as np

# Class labels used by the trained YOLOv8 model
CLASSES = {
    0: "Normal",
    1: "Pneumonia",
    2: "TB",
}


def load_model(weights_path: str) -> YOLO:
    """
    Load the trained YOLOv8 classification model.

    Args:
        weights_path (str):
            Path to the .pt model weights file.

    Returns:
        YOLO:
            Loaded YOLO model ready for inference.
    """
    model = YOLO(weights_path)
    return model


def predict_image(model: YOLO, image: Image.Image) -> dict:
    """
    Run prediction on a chest X-ray image.

    The image is converted to RGB format before inference.
    The function returns:
        - predicted disease class
        - class index
        - confidence score
        - probability scores for all classes

    Args:
        model (YOLO):
            Loaded YOLOv8 classification model.

        image (PIL.Image.Image):
            Input chest X-ray image.

    Returns:
        dict:
            Structured prediction results.
    """

    # Convert image into RGB numpy array
    image_array = np.array(image.convert("RGB"))

    # Run model inference
    results = model(image_array, verbose=False)

    # Extract probability results
    probabilities = results[0].probs

    if probabilities is None:
        raise ValueError(
            "Model did not return classification probabilities."
        )

    # Get highest-confidence prediction
    predicted_index = int(probabilities.top1)
    confidence_score = float(probabilities.top1conf)

    # Get probabilities for all classes
    probability_values = probabilities.data.tolist()

    # Build formatted response
    prediction_result = {
        "predicted_class": CLASSES.get(predicted_index, "Unknown"),
        "class_index": predicted_index,
        "confidence": round(confidence_score * 100, 2),
        "probabilities": {
            CLASSES[i]: round(float(score) * 100, 2)
            for i, score in enumerate(probability_values)
        },
    }

    return prediction_result