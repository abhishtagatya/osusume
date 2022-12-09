import nltk
import pandas as pd

nltk.download('wordnet')
nltk.download('omw-1.4')

from osusume import Osusume
from osusume.util.text import Remove, Transform


def text_clean_str(text: str):
    text = Remove.numbers(text)
    text = Remove.punctuations(text)
    text = Remove.whitespaces(text)
    text = Remove.capitalfirsts(text)
    text = Transform.lemma(text, 'WORDNET')
    text = text.lower()
    return text


def text_clean_frame(dataframe: pd.DataFrame, text_col: str):
    dataframe[text_col] = dataframe[text_col].apply(Remove.numbers)
    dataframe[text_col] = dataframe[text_col].apply(Remove.punctuations)
    dataframe[text_col] = dataframe[text_col].apply(Remove.whitespaces)
    dataframe[text_col] = dataframe[text_col].apply(Remove.capitalfirsts)
    dataframe[text_col] = dataframe[text_col].apply(Remove.specifics, args=('written by mal rewrite',))
    dataframe[text_col] = dataframe[text_col].apply(Transform.lemma, args=('WORDNET',))
    dataframe[text_col] = dataframe[text_col].str.lower()
    return dataframe


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = '../data/'
    df_anime = pd.read_json(data_dir + 'anime_dataset.json')

    df_top_anime = df_anime[
        (df_anime['synopsis'].notnull()) &
        (df_anime['mal_popularity'] != 0) &
        (df_anime['type'] == 'TV')
        ].sort_values(by='mal_popularity')

    df_top_anime = df_top_anime[
                       (df_top_anime['mal_popularity'] < 5000) &
                       (df_top_anime['mal_rank'] < 5000)
                       ].reset_index(drop=True)[:250]

    df_top_anime = df_top_anime[[
        'title', 'synopsis', 'type', 'aired_start', 'aired_end',
        'mal_rating', 'mal_score', 'mal_reviewer', 'mal_rank', 'mal_popularity']
    ].dropna(subset=['synopsis'])
    df_top_anime['synopsis_clean'] = df_top_anime['synopsis']
    df_top_anime.to_json(data_dir + 'anime_sm.json', orient='records')

    df_top_anime = text_clean_frame(
        dataframe=df_top_anime,
        text_col='synopsis'
    )

    engine = Osusume.from_dict({
        'token': '<cohere-token>',
        'dataset': df_top_anime,
        'sentiment_column': 'synopsis',
        'response_prompt': data_dir + 'anime_data.txt',
        'response_column': [
            ('Title', 'title'),
            ('Synopsis', 'synopsis')
        ]
    }).fit()

    print(engine.predict('<enter-your-query>', n_out=1))
