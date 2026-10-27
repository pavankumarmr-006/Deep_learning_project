
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Text cannot be empty")
        return value


class BatchTicketRequest(BaseModel):
    tickets: list[TicketRequest] = Field(
        ..., min_length=1, max_length=100
    )


class TicketPredictor:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path)
        ).to(DEVICE)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(DEVICE)

        with torch.inference_mode():
            probabilities = torch.softmax(
                self.model(**inputs).logits, dim=-1
            )[0]

        index = int(probabilities.argmax().item())
        label = self.model.config.id2label.get(
            index,
            self.model.config.id2label.get(str(index), str(index))
        )
        return str(label), float(probabilities[index].item())


def load_models():
    return {
        "category": TicketPredictor(
            MODEL_DIR / "distilbert_category"
        ),
        "priority": TicketPredictor(
            MODEL_DIR / "distilbert_priority"
        ),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictors = load_models()
    yield


app = FastAPI(
    title="Customer Support Ticket Classifier",
    lifespan=lifespan
)


def classify(text, predictors):
    category, cat_conf = predictors["category"].predict(text)
    priority, pri_conf = predictors["priority"].predict(text)

    return {
        "category": category.title(),
        "priority": priority.title(),
        "confidence": round((cat_conf + pri_conf) / 2, 4)
    }


@app.get("/health")
def health(request: Request):
    ready = len(getattr(request.app.state, "predictors", {})) == 2
    return {
        "status": "ok" if ready else "not_ready",
        "models_loaded": ready
    }


@app.post("/predict")
def predict(payload: TicketRequest, request: Request):
    try:
        return classify(payload.text, request.app.state.predictors)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Prediction failed"
        ) from exc


@app.post("/predict-batch")
def predict_batch(payload: BatchTicketRequest, request: Request):
    try:
        predictions = [
            classify(ticket.text, request.app.state.predictors)
            for ticket in payload.tickets
        ]
        return {"predictions": predictions}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Batch prediction failed"
        ) from exc
