from fastapi import FastAPI, Depends

from src.dependencies import user_active, user_authenticate
from src.security.token import create_access_token
from src.dtypes import PredictionInMulti, User
from src.bl import BusinessLogic
from config import ACCESS_TOKEN_TYPE


app = FastAPI()


@app.post('/token')
async def login_for_access_token(
        user: User = Depends(user_authenticate),
):
    # Create token for user
    access_token = create_access_token({'sub': user.username})

    return {
        'access_token': access_token,
        'token_type': ACCESS_TOKEN_TYPE,
    }


@app.post('/users/me')
async def get_current_user(
        user: User = Depends(user_active),
):
    return {
        'user': user,
    }


@app.post('/predict')
async def predict(
        request: PredictionInMulti,
):
    return BusinessLogic.request_predict(request)
