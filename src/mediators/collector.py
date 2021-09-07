import torch
from src.modules.collector import Collector
from src.dtypes import Model


class MediatorCollector:
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
