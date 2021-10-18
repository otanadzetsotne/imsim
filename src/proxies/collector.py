import torch
from src.utils.collector import Collector
from src.dtypes import Model


class ProxyCollector:
    @classmethod
    def collect(
            cls,
            model_type: Model,
    ) -> torch.nn.Module:
        """
        Create model objects
        :param model_type: Model
        :return: torch.nn.Module
        """

        return Collector.collect(model_type)
