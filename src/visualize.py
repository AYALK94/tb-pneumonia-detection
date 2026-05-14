from PIL import Image, ImageDraw
import io
import base64

# Color scheme for each prediction class
# Used to visually distinguish results on the annotated image
COLOR_MAP = {
    "Normal": (0, 180, 0),
    "Pneumonia": (255, 140, 0),
    "TB": (210, 0, 0),
}


def draw_prediction(image: Image.Image, prediction: dict) -> str:
    """
    Draw prediction results on a chest X-ray image and return it as a base64 string.

    The function:
    - Resizes the image for consistent display
    - Draws a label bar with predicted class and confidence
    - Encodes the final image as a base64 data URL for web use

    Args:
        image (PIL.Image.Image):
            Original X-ray image.

        prediction (dict):
            Model output containing:
            - predicted_class (str)
            - confidence (float)

    Returns:
        str:
            Base64-encoded image string (data URL format)
    """

    # Create a copy of the image to avoid modifying original
    annotated_image = image.copy().resize((400, 400))

    draw = ImageDraw.Draw(annotated_image)

    # Extract prediction details
    label = prediction.get("predicted_class", "Unknown")
    confidence = prediction.get("confidence", 0)
    color = COLOR_MAP.get(label, (200, 200, 200))

    # Draw top label bar
    draw.rectangle([0, 0, 400, 44], fill=(20, 20, 20))

    # Write prediction text
    draw.text((10, 12), f"{label}  {confidence}%", fill=color)

    # Convert image to base64 string
    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG")

    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded_image}"