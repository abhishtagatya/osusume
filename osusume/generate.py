from typing import List

import pandas as pd

from cohere.client import Client


class Generate:

    def __init__(self,
                 cohere_obj: Client,
                 response_prompt: str,
                 response_column: List,
                 response_separator: str = '--',
                 model: str = 'large',
                 n_set: int = 5
                 ):
        self._cohere_obj = cohere_obj
        self._response_prompt = response_prompt
        self._response_column = response_column
        self._response_separator = response_separator
        self._model = model

        self._sliced_list = self._response_prompt.split(self._response_separator)[-n_set:]
        self._sliced = '--'.join(self._sliced_list)

    def generate(self,
                 item,
                 n_set: int = 5,
                 max_token: int = 50,
                 temperature: float = 0.9,
                 k: int = 0,
                 p: float = 0.75,
                 frequency_penalty: int = 0,
                 presence_penalty: int = 0,
                 return_likelihoods: str = 'NONE'
                 ):
        if len(self._sliced_list) != n_set:
            self._sliced_list = self._response_prompt.split(self._response_separator)[-n_set:]
            self._sliced = '--'.join(self._sliced_list)

        query_prompt = self._sliced + "--\n"
        for p_name, p_col in self._response_column:
            query_prompt += "{name}: {col}\n".format(name=p_name, col=item[p_col])
        query_prompt += f"Recommend Text: "

        response = self._cohere_obj.generate(
            model=self._model,
            prompt=query_prompt,
            max_tokens=max_token,
            temperature=temperature,
            k=k,
            p=p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=[self._response_separator],
            return_likelihoods=return_likelihoods
        )
        return response.generations[0].text.replace('--', '').replace('\n', '')
