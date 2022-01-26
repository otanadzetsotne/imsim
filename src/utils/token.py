from datetime import datetime
from datetime import timedelta

from jose import jwt


def create_access_token(
        data: dict,
        token_algorithm: str,
        token_expires: timedelta,
        token_secret_key: str,
) -> str:
    """
    Create jwt token string with encoded data
    """

    # Prepare data
    to_encode = data.copy()
    expire = datetime.utcnow() + token_expires
    to_encode.update({'exp': expire})

    # Create JWT
    encoded_jwt = jwt.encode(
        claims=to_encode,
        key=token_secret_key,
        algorithm=token_algorithm,
    )

    return encoded_jwt
