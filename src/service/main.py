import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncGenerator

from fastapi import FastAPI, Query
from loaded_models import load_item_features, load_model
from schemas import PredictPurchaseErrorResponse, PredictPurchaseRequest, PredictPurchaseResponse


# Возможно надо еще импортировать prometheus, pydantic (схемы)
# from schemas import PredictPurchaseRequest, PredictPurchaseResponse, ErrorResponse
# from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app_state = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, Any]:
    app_state["ranker"] = load_model()
    app_state["item_features"] = load_item_features()

    yield

    app_state.clear()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "0.0.1"

app = FastAPI(
    title="Retail Recommendation API",
    version=APP_VERSION,
    description="API для выдачи рекомендаций и вероятностей покупки.",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
def health_check():
    """
    Проверка состояния сервиса и доступности моделей.
    """

    ok = {
        "ranker_loaded": app_state.get("ranker") is not None,
        "items_in_feature_store": len(app_state.get("item_features", [])),
    }
    logger.info(f"Health check: {ok}")
    return {"status": "ok", "details": ok}


@app.get("/version", tags=["system"])
def version():
    """
    Возвращает текущую версию сервиса.
    """
    return {"version": APP_VERSION, "service": "retail_recsys"}


@app.get("/", tags=["system"])
def root():
    return {"message": "Retail recommender API running", "version": APP_VERSION}


@app.get("/predict_purchase", response_model=PredictPurchaseResponse)
def predict_purchase(
    params: Annotated[PredictPurchaseRequest, Query()],
) -> PredictPurchaseResponse | PredictPurchaseErrorResponse:
    """Предсказываем вероятность покупки товара

    Note:
        Предсказания вероятности покупки товара в конкретный день недели и время.

    Args:
        params: PredictPurchaseRequest
            item_id: номер товара в базе данных.
            hour: час текущий.
            weekday: день недели текущий.
    """
    item_features = app_state["item_features"]
    ranker = app_state["ranker"]

    row = item_features[item_features["itemid"] == params.item_id].copy()
    if row.empty:
        return PredictPurchaseErrorResponse(error="item not found")

    row["hour"] = params.hour
    row["weekday"] = params.weekday
    x = row[["views", "purchases", "ctr", "hour", "weekday", "categoryid", "available"]].astype(float)
    prob = float(ranker.predict_proba(x)[0][1])

    return PredictPurchaseResponse(item_id=params.item_id, purchase_probability=prob)


@app.get("/metrics", tags=["system"])
def metrics():
    """Надо как-то все логировать"""
    # Заполни меня - основная логика по которой мы отдаем метрики в prometheus.
    ...


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=32000, workers=1)
