from fastapi import FastAPI
from src.dtypes import PredictionInMulti
from src.bl import BusinessLogic


app = FastAPI()


@app.post('/predict')
async def predict(
        request: PredictionInMulti,
):
    return BusinessLogic.request_predict(request)
