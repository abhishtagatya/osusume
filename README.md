# Osusume (おすすめ)

> Text-Based Recommendation Engine using Cohere

Osusume is a library to implement text-based recommendation engine using Cohere Embeddings and Generation API. 
Initally was built for a Hacktahon Submission hosted by Lablab.AI and Cohere.

[View Changelog](github.com/abhishtagatya/osusume/blob/master/CHANGELOG.md)

## Quick Start

```python
import pandas as pd

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from osusume import Osusume

anime_df = pd.read_json('anime_dataset.json')

engine = Osusume.from_dict({
    'token': '<cohere-token>',
    'dataset': anime_df,
    'sentiment_column': 'synopsis',
    'response_prompt': 'anime_data.txt',
    'response_column': [
        ('Title', 'title'),
        ('Synopsis', 'synopsis')
    ]
}).fit()

engine.predict(
    query='animes about monsters and aliens from another world', 
    n_out=3
)

# Top 3 Anime's from given Query

```

## Installation

```bash
~ pip install osusume --upgrade
```
You need to use Pip to install osusume. Conda package is currently unavailable.

### Requirements
* Python >= 3.8
* Cohere
* Pandas
* Numpy
* Scikit Learn
* NLTK

## Author
* Abhishta Gatya ([Email](mailto:abhishtagatya@yahoo.com)) - Software and Machine Learning Engineer