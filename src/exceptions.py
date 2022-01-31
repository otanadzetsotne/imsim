from fastapi import HTTPException, status
from requests.models import HTTPError


class BadUrlError(HTTPError):
    """
    Raises when url is invalid
    """

    code = 400


class NeuralNetworkModelNotFoundError(Exception):
    """
    Raises when neural network model not found
    """
    pass


# Authentication errors


class UsernameOrPasswordError(HTTPException):
    def __init__(self):
        super(UsernameOrPasswordError, self).__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Incorrect username or password',
            headers={'WWW-Authenticate': 'Bearer'},
        )


class CredentialsError(HTTPException):
    def __init__(self):
        super(CredentialsError, self).__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )


class InactiveUserError(HTTPException):
    def __init__(self):
        super(InactiveUserError, self).__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Inactive user',
        )
