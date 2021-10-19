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
