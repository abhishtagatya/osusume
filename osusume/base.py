import os.path
import pathlib
import json
import yaml
from yaml.loader import SafeLoader
from typing import List, Dict

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from cohere.client import Client


class BaseEngine:

    SIMILARITY_MODEL = [
        'CSI'
    ]

    def __init__(self,
                 token: str,
                 dataset: pd.DataFrame,
                 columns: List = None,
                 sentiment_column: str = '',
                 response_prompt: str = '',
                 response_column: List = None,
                 response_separator: str = '',
                 embedding_model: str = 'large',
                 generation_model: str = 'large',
                 similarity_model: str = 'CSI'
                 ):
        self.sentiment_column = sentiment_column
        self.dataset = dataset

        self._token = token
        self._columns = columns
        self._similarity_model = similarity_model

        self.response_prompt = response_prompt
        self.response_column = response_column
        self.response_separator = response_separator

        self.cohere = Client(self._token)
        self.embedding_model = embedding_model
        self.generation_model = generation_model

    @classmethod
    def from_dict(cls, config: Dict = None):
        """
        Load Configuration from Dictionary Object
        :param config: Dict
        :return: Instance
        """
        if config is None:
            raise ValueError('Config cannot be empty or None')

        token = config.get('token', '')
        if token == '':
            raise ValueError('Config key `token` cannot be empty.')

        dataset = config.get('dataset', '')
        if type(dataset) is str:
            if os.path.exists(dataset):
                ext_file_df = pathlib.Path(dataset).suffix
                if ext_file_df == '.csv':
                    dataset = pd.read_csv(dataset)
                if ext_file_df == '.json':
                    dataset = pd.read_json(dataset)
                if ext_file_df == '.xml':
                    dataset = pd.read_xml(dataset)
        if type(dataset) is not pd.DataFrame:
            raise ValueError('Config key `dataset` is not `pd.Dataframe` nor `str` object.')

        columns = config.get('columns', dataset.columns)

        sentiment_column = config.get('sentiment_column', columns[0])
        if sentiment_column not in columns:
            raise ValueError('Config key `sentiment_column` is not in `columns` list.')

        response_prompt = config.get('response_prompt', '')
        if os.path.exists(response_prompt):
            with open(response_prompt, 'r') as prompt_file:
                response_prompt = prompt_file.read()

        response_column = config.get('response_column', [(col, col) for col in columns])
        response_separator = config.get('response_separator', '--')
        embedding_model = config.get('embedding_model', 'large')
        generation_model = config.get('generation_model', 'large')
        similarity_model = config.get('similarity_model', BaseEngine.SIMILARITY_MODEL[0])
        if similarity_model not in BaseEngine.SIMILARITY_MODEL:
            raise NotImplementedError(f'Config key `similarity_model` of `{similarity_model}` is not implemented.')

        return cls(
            token=token,
            dataset=dataset,
            columns=columns,
            sentiment_column=sentiment_column,
            response_prompt=response_prompt,
            response_column=response_column,
            response_separator=response_separator,
            embedding_model=embedding_model,
            generation_model=generation_model,
            similarity_model=similarity_model
        )

    @classmethod
    def from_json(cls, config: str = None):
        if os.path.exists(config):
            ext_file_df = pathlib.Path(config).suffix
            if ext_file_df == '.json':
                with open(config, 'r') as config_file:
                    config_json = json.load(config_file)

                return cls.from_dict(config_json)
            else:
                raise ValueError(f'Config file {config} is not in JSON Format.')

        raise FileNotFoundError(f'Config file {config} is not found.')

    @classmethod
    def from_yaml(cls, config: str = None):
        if os.path.exists(config):
            ext_file_df = pathlib.Path(config).suffix
            if ext_file_df == '.yaml' or ext_file_df == '.yml':
                with open(config, 'r') as config_file:
                    config_yaml = yaml.load(config_file, Loader=SafeLoader)

                return cls.from_dict(config_yaml)
            else:
                raise ValueError(f'Config file {config} is not in YAML Format.')

        raise FileNotFoundError(f'Config file {config} is not found.')

    def calculate_sim(self, emb_a, emb_b, n_out: int = 5):
        if self._similarity_model is None or self._similarity_model not in self.SIMILARITY_MODEL:
            raise NotImplementedError('Config key `similarity_model` of `{similarity_model}` is not implemented.')

        if self._similarity_model == 'CSI':
            """
                Cosine Similarity Index
            """
            cosine_sim = cosine_similarity(emb_a, emb_b)
            sim_scores = list(enumerate(cosine_sim[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[:n_out]
            indices = [i[0] for i in sim_scores]
            return indices

        return
