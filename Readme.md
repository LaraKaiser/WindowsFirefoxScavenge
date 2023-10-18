# Analyse browising habits - Windows
Firefox browsing history tool to quickly access daily data and build an interest timeline of a user for Windows.
Automatically trains model with a given data-set and safes it.

## Prerequirements
**Gensim**<br/>
For the model
```
pip install gensim
```
**NLTK**
```
pip install nltk
```
And download [NLTK Data](https://www.nltk.org/data.html) for processing
the search queries. English and German stopwords.

**pyLDAvis** <br/>
For visualisation
```
pip install pyLDAvis
```
## Training Data
For the model I used these [Search Engine Keywords](https://www.kaggle.com/datasets/hofesiy/2019-search-engine-keywords) and had a Coherence Score with a 10th of the data of 0.7.
Depending on which topic you are screening for, create your own dictionary by creating a json with data structured as such: 
```
{"id":"keyphrase":"SEARCH-QUERY"}

```

## How to use it
```
python <File> <Start-Date> <End-Date> <PATH-TO-TRAIN-DATA>
python .\seachQueryAnalysis.py 2023-08-09 2023-09-19 C:\\Users\\xxx\\Downloads\\keyphrases.json
```

## TODO
- Get a better Model: Mallet Model
- Variety of histories and crosslinks
- ...