import numpy as np
import pandas as pd
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class FeatureExtract:

    def __init__(self, series: pd.Series, language: str = 'english'):
        self._series = series
        self._language = language

        self.algorithm = TfidfVectorizer(stop_words=self._language)
        self.algorithm.fit_transform(self._series)

    def extract(self, text: str, limit: int = 5) -> List[str]:
        _feat_ext = self.algorithm.transform([text])
        _feat_array = np.array(self.algorithm.get_feature_names())
        _feat_sort = np.argsort(_feat_ext.toarray()).flatten()[::-1]

        return _feat_array[_feat_sort][:limit]
