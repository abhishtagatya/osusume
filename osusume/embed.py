import pandas as pd

from cohere.client import Client


class Embedding:

    def __init__(self,
                 series: pd.Series,
                 cohere_obj: Client,
                 model: str = 'large'):
        self._series = series
        self._cohere_obj = cohere_obj
        self._model = model

    def embed(self):
        cohere_resp = self._cohere_obj.embed(
            texts=self._series.tolist(),
            model=self._model
        )
        return cohere_resp.embeddings
