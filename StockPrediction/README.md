# Two Sigma: Using News to Predict Stock Movements

## Notes
**Keras Embedding Parameters:**
- input_dim : the vocabulary size. This is how many unique words are represented in your corpus.

- output_dim : the desired dimension of the word vector. For example, if output_dim = 100, then every word will be mapped onto a vector with 100 elements, whereas if output_dim = 300, then every word will be mapped onto a vector with 300 elements.

- input_length : the length of your sequences. For example, if your data consists of sentences, then this variable represents how many words there are in a sentence. As disparate sentences typically contain different number of words, it is usually required to pad your sequences such that all sentences are of equal length. The keras.preprocessing.pad_sequence method can be used for this (https://keras.io/preprocessing/sequence/).

## LSTM Configurations

**Market Features:**
```python
time_cols = [‘year’, ‘week’, ‘day’, ‘dayofweek’]

numeric_cols = ['volume', 'close', 'open', 
'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']

[‘assetCode_encoded’]
```

**News Features:**
```python
news_cols_agg = {
    'urgency': ['min', 'count'],
    'takeSequence': ['max'],
    'bodySize': ['min', 'max', 'mean', 'std'],
    'wordCount': ['min', 'max', 'mean', 'std'],
    'sentenceCount': ['min', 'max', 'mean', 'std'],
    'companyCount': ['min', 'max', 'mean', 'std'],
    'marketCommentary': ['min', 'max', 'mean', 'std'],
    'relevance': ['min', 'max', 'mean', 'std'],
    'sentimentNegative': ['min', 'max', 'mean', 'std'],
    'sentimentNeutral': ['min', 'max', 'mean', 'std'],
    'sentimentPositive': ['min', 'max', 'mean', 'std'],
    'sentimentWordCount': ['min', 'max', 'mean', 'std'],
    'noveltyCount12H': ['min', 'max', 'mean', 'std'],
    'noveltyCount24H': ['min', 'max', 'mean', 'std'],
    'noveltyCount3D': ['min', 'max', 'mean', 'std'],
    'noveltyCount5D': ['min', 'max', 'mean', 'std'],
    'noveltyCount7D': ['min', 'max', 'mean', 'std'],
    'volumeCounts12H': ['min', 'max', 'mean', 'std'],
    'volumeCounts24H': ['min', 'max', 'mean', 'std'],
    'volumeCounts3D': ['min', 'max', 'mean', 'std'],
    'volumeCounts5D': ['min', 'max', 'mean', 'std'],
    'volumeCounts7D': ['min', 'max', 'mean', 'std']
    }
```

**Preprocessing:**

- ‘bfill’ missing data
- Remove outliers that fall out of [-0.2, 0.2]
- Remove cases with close/open > 2
- Both market and news fit with StandardScaler()
- Merge market and news by [‘time’]

**Model:**
```python
if toy:
    batch_size=1000
    validation_batch_size=1000
    steps_per_epoch=5
    validation_steps=2
    epochs=5
    ModelFactory.look_back=30
    ModelFactory.look_back_step=5

else:
    batch_size=1000
    validation_batch_size=1000
    steps_per_epoch=20
    validation_steps=5
    epochs=20

model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=64, return_sequences=True)) model.add(LSTM(units=32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

earlystopper = EarlyStopping(patience=5, verbose=1)
```
