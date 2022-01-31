from jose import JWTError, jwt
from fastapi import Depends
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)

from src.security.password import PasswordContext
from src.dependencies.settings import (
    Settings,
    get_settings,
)
from src.dtypes import (
    User,
    UserDB,
    TokenData,
)
from src.exceptions import (
    InactiveUserError,
    UsernameOrPasswordError,
    CredentialsError,
)


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl='token',
)


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
def get_user(
        username: str,
) -> UserDB:
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserDB(**user_dict)


async def user_token_valid(
        token: str = Depends(oauth2_scheme),
        settings: Settings = Depends(get_settings),
) -> User:
    """
    Get current user from jwt token
    """

    try:
        # Decode data from token
        payload = jwt.decode(
            token=token,
            key=settings.secret.key_token,
            algorithms=[settings.token.algorithm],
        )
        # Store username in token subject
        username: str = payload.get('sub')

        if username is None:
            raise CredentialsError

        token_data = TokenData(username=username)

    except JWTError:
        raise CredentialsError

    # Get user entity
    user = get_user(username=token_data.username)

    if not user:
        raise CredentialsError

    return user


async def user_active(
        user: User = Depends(user_token_valid),
) -> User:
    """
    Check if current user is activated
    :param user: User object
    """

    if user.disabled:
        raise InactiveUserError

    return user


async def user_authenticate(
        form_data: OAuth2PasswordRequestForm = Depends(),
        pwd_context: PasswordContext = Depends(),
) -> User:
    """
    Check if user credentials are correct
    """

    user = get_user(form_data.username)

    if not user:
        raise UsernameOrPasswordError

    if not pwd_context.verify(form_data.password, user.password_hash):
        raise UsernameOrPasswordError

    return user
