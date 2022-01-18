from datetime import datetime

from jose import jwt

from config import SECRET_KEY
from config import ACCESS_TOKEN_ALGORITHM
from config import ACCESS_TOKEN_EXPIRES


def create_access_token(
        data: dict,
) -> str:
    """
    Create jwt token string with encoded data
    :param data: Data to encode
    """

    # Prepare data
    to_encode = data.copy()
    expire = datetime.utcnow() + ACCESS_TOKEN_EXPIRES
    to_encode.update({'exp': expire})

    # Create JWT
    encoded_jwt = jwt.encode(
        claims=to_encode,
        key=SECRET_KEY,
        algorithm=ACCESS_TOKEN_ALGORITHM,
    )

    return encoded_jwt
