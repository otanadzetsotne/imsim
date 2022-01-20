from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from src.security.password import PasswordContext
from src.dtypes import User
from src.dtypes import UserDB
from src.dtypes import TokenData
from config import SECRET_KEY
from config import ACCESS_TOKEN_ALGORITHM


# TODO: удалить
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "password_hash": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
    },
}


# TODO: унести
def get_user(username: str):
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserDB(**user_dict)


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl='token',
)
pwd_context = PasswordContext()


async def user_token_valid(
        token: str = Depends(oauth2_scheme),
) -> User:
    """
    Get current user from jwt token
    :param token: string of token
    """

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Could not validate credentials',
        headers={'WWW-Authenticate': 'Bearer'},
    )

    try:
        # Decode data from token
        payload = jwt.decode(
            token=token,
            key=SECRET_KEY,
            algorithms=[ACCESS_TOKEN_ALGORITHM],
        )
        # Store username in token subject
        username: str = payload.get('sub')

        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username)

    except JWTError:
        raise credentials_exception

    # Get user entity
    user = get_user(username=token_data.username)

    if not user:
        raise credentials_exception

    return user


async def user_active(
        user: User = Depends(user_token_valid),
) -> User:
    """
    Check if current user is activated
    :param user: User object
    """

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Inactive user',
        )

    return user


async def user_authenticate(
        form_data: OAuth2PasswordRequestForm = Depends(),
) -> User:
    """
    Check if user credentials are correct
    :param form_data: OAuth2PasswordRequestForm
    """

    exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Incorrect username or password',
        headers={'WWW-Authenticate': 'Bearer'},
    )

    user = get_user(form_data.username)

    if not user:
        raise exception

    if not pwd_context.verify(form_data.password, user.password_hash):
        raise exception

    return user


# TODO: We need to create whole user requirements dependency hierarchy:
#  + Roles
#  + Dependency classes
