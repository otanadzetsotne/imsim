from fastapi import FastAPI, Depends

from config import Settings
from src.dtypes import PredictionInMulti, User
from src.bl import BusinessLogic
from src.utils.token import create_access_token
from src.dependencies import (
    get_settings,
    user_active,
    user_authenticate,
)


app = FastAPI()


@app.post('/token')
async def login_for_access_token(
        user: User = Depends(user_authenticate),
        settings: Settings = Depends(get_settings),
):
    # Create token for user
    access_token = create_access_token(
        {'sub': user.username},
        settings.token.algorithm,
        settings.token.expires,
        settings.secret.key_token,
    )

    return {
        'access_token': access_token,
        'token_type': settings.token.type,
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
