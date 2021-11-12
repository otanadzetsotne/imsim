# local
from src.proxies.downloader import ProxyDownloader
from src.proxies.predictor import ProxyPredictor
from src.proxies.collector import ProxyCollector
from src.proxies.images import ProxyImages


class Proxies:
    """
    Proxies facade class that specifies high level contracts for
    functions given / received data types
    """

    downloader = ProxyDownloader
    predictor = ProxyPredictor
    collector = ProxyCollector
    images = ProxyImages
