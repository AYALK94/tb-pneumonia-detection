from PIL import Image
import io
import os

# Supported image formats for the model
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Maximum allowed upload size (in megabytes)
MAX_FILE_SIZE_MB = 10


def validate_image(filename: str, file_bytes: bytes) -> bool:
    """
    Check whether an uploaded image is valid for inference.

    This function ensures:
    - The file extension is supported
    - The file size does not exceed the allowed limit

    Args:
        filename (str): Name of the uploaded file
        file_bytes (bytes): Raw file content

    Returns:
        bool: True if the file is valid

    Raises:
        ValueError: If the file type or size is invalid
    """

    file_extension = os.path.splitext(filename)[-1].lower()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_extension}")

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File too large: {file_size_mb:.1f} MB "
            f"(maximum allowed is {MAX_FILE_SIZE_MB} MB)"
        )

    return True


def bytes_to_image(file_bytes: bytes) -> Image.Image:
    """
    Convert raw image bytes into a PIL Image in RGB format.

    Args:
        file_bytes (bytes): Raw uploaded image bytes

    Returns:
        Image.Image: RGB image ready for model inference
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return image


def get_confidence_label(confidence: float) -> str:
    """
    Convert a numeric confidence score into a human-readable label.

    Args:
        confidence (float): Confidence score (0–100 scale)

    Returns:
        str: Label representing confidence level
    """

    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 50:
        return "Moderate"
    else:
        return "Low"