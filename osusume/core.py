import warnings
from typing import Any, List, Tuple

import pandas as pd

from osusume.base import BaseEngine
from osusume.extract import FeatureExtract
from osusume.embed import Embedding
from osusume.generate import Generate
from osusume.util.text import Remove, Transform


def clean_text(text: str):
    text = Remove.numbers(text)
    text = Remove.punctuations(text)
    text = Remove.whitespaces(text)
    text = Remove.capitalfirsts(text)
    text = Transform.lemma(text, 'WORDNET')
    text = text.lower()
    return text


class Osusume(BaseEngine):

    def __init__(self,
                 token: str,
                 dataset: pd.DataFrame,
                 columns: List = None,
                 sentiment_column: str = '',
                 response_prompt: str = '',
                 response_column: str = '',
                 response_separator: str = '',
                 embedding_model: str = 'large',
                 generation_model: str = 'large',
                 similarity_model: str = 'CSI'
                 ):
        super().__init__(token, dataset, columns, sentiment_column, response_prompt, response_column,
                         response_separator, embedding_model, generation_model, similarity_model)
        self._dataset_emb = None
        self._generator = None

        self._feat_ext_col = f'{__name__}-keyword'
        self._feat_ext_col2 = f'{__name__}-keyword_str'

    def fit(self):
        if self.dataset[self.sentiment_column].isnull().values.any():
            raise ValueError(f'Found `NaN` values at `sentiment_column` of {self.sentiment_column}')

        _feature_ext = FeatureExtract(
            series=self.dataset[self.sentiment_column]
        )

        self.dataset[self._feat_ext_col] = self.dataset[self.sentiment_column].apply(_feature_ext.extract)
        self.dataset[self._feat_ext_col2] = self.dataset[self._feat_ext_col].apply(lambda x: ' '.join(x))

        self.response_column.append(('Keyword', self._feat_ext_col2))

        _embedder = Embedding(
            series=self.dataset[self._feat_ext_col2],
            cohere_obj=self.cohere,
            model=self.embedding_model
        )

        self._dataset_emb = _embedder.embed()

        self._generator = Generate(
            cohere_obj=self.cohere,
            response_prompt=self.response_prompt,
            response_column=self.response_column,
            response_separator=self.response_separator,
            model=self.generation_model
        )

        return self

    def predict(self, query: str, with_adjustment: Tuple[str, str] = None, n_out: int = 5):
        if self._dataset_emb is None:
            raise AttributeError(f'Variable `{__name__}._dataset_emb` is None.')

        if self._generator is None:
            raise AttributeError(f'Variable `{__name__}._generator` is None.')

        query = clean_text(query)
        query_emb = Embedding(
            series=pd.Series([query]),
            cohere_obj=self.cohere,
            model=self.embedding_model
        ).embed()

        rec_indices = self.calculate_sim(
            emb_a=query_emb,
            emb_b=self._dataset_emb,
            n_out=n_out
        )

        df_relocate = self.dataset.iloc[rec_indices]

        if with_adjustment:
            column, order = with_adjustment
            if order == 'ASC':
                df_relocate = df_relocate.sort_values(by=[column], ascending=True)
            else:
                df_relocate = df_relocate.sort_values(by=[column], ascending=False)

        recommend_text = []
        for _, recommendation in df_relocate.iterrows():
            recommend_text.append(self._generator.generate(recommendation))
        return recommend_text


