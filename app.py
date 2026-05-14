from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn, os

from src.predict   import load_model, predict_image
from src.utils     import validate_image, bytes_to_image, get_confidence_label
from src.visualize import draw_prediction

app = FastAPI(title="TB & Pneumonia Detection API", version="1.0.0")

os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates  = Jinja2Templates(directory="templates")
MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
model      = load_model(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,       # ← keyword argument
        name="index.html"      # ← keyword argument
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        validate_image(file.filename, file_bytes)
        image      = bytes_to_image(file_bytes)
        prediction = predict_image(model, image)

        prediction["confidence_label"] = get_confidence_label(prediction["confidence"])
        prediction["annotated_image"]  = draw_prediction(image, prediction)
        return JSONResponse(content=prediction)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)