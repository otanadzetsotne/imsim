from src.mediators.downloader import MediatorDownloader
from src.mediators.predictor import MediatorPredictor
from src.mediators.collector import MediatorCollector
from src.mediators.images import MediatorImages


class MediatorFacade:
    """
    Mediator facade class that specifies high level contracts for functions given / received data types
    """

    downloader = MediatorDownloader
    predictor = MediatorPredictor
    collector = MediatorCollector
    images = MediatorImages
