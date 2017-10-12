
## DataSchool - Machine Learning with Text and Python:  

### Week 6 Self Imposed Homework

I am going to use the Kaggle competition that inspired me to sign up for this course.  Using what I have learned in the Machine Learning with Text course, I will attempt to beat the scores on the leader board.

The Kaggle competition is:

![SA Image](sa_emotions_picture.png) 

### [Sentiment Analysis: Emotion in Text.](https://www.kaggle.com/c/sa-emotions)

*Identify emotion in text using sentiment analysis.*


```python
import numpy as np
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from stemming.porter2 import stem
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline, make_union
from sentiment_analysis.transformers import RemoveEllipseTransformer, RemoveHtmlEncodedTransformer, RemoveNumbersTransformer, RemoveSpecialCharactersTransformer, RemoveUsernameTransformer, RemoveUrlsTransformer
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.base import TransformerMixin
import re
import nltk.stem
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from nltk.sentiment.util import mark_negation
```

    /Users/youngsoul/Documents/Development/PythonDev/VirtualEnvs/MLText2Env/lib/python3.6/site-packages/nltk/twitter/__init__.py:20: UserWarning: The twython library has not been installed. Some functionality from the twitter package will not be available.
      warnings.warn("The twython library has not been installed. "



```python
# allow plots to appear in the notebook
%matplotlib inline
```


```python
# read the training data
training_data = pd.read_csv('../data/kaggle/sa-emotions/train_data_lexicon.csv')
```


```python
training_data.shape
```




    (30000, 7)




```python
training_data.head(20)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>sentiment</th>
      <th>content</th>
      <th>disposition</th>
      <th>neg_words</th>
      <th>neu_words</th>
      <th>pos_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>empty</td>
      <td>@tiffanylue i know  i was listenin to bad habi...</td>
      <td>-1.0</td>
      <td>0.050000</td>
      <td>0.950000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>sadness</td>
      <td>Layin n bed with a headache  ughhhh...waitin o...</td>
      <td>-1.0</td>
      <td>0.076923</td>
      <td>0.923077</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>sadness</td>
      <td>Funeral ceremony...gloomy friday...</td>
      <td>-1.0</td>
      <td>0.166667</td>
      <td>0.833333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>enthusiasm</td>
      <td>wants to hang out with friends SOON!</td>
      <td>-1.0</td>
      <td>0.125000</td>
      <td>0.875000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>neutral</td>
      <td>@dannycastillo We want to trade with someone w...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>worry</td>
      <td>Re-pinging @ghostridah14: why didn't you go to...</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.950000</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>sadness</td>
      <td>I should be sleep, but im not! thinking about ...</td>
      <td>-1.0</td>
      <td>0.058824</td>
      <td>0.941176</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>worry</td>
      <td>Hmmm. http://www.djhero.com/ is down</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>sadness</td>
      <td>@charviray Charlene my love. I miss you</td>
      <td>-1.0</td>
      <td>0.125000</td>
      <td>0.875000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>sadness</td>
      <td>@kelcouch I'm sorry  at least it's Friday?</td>
      <td>-1.0</td>
      <td>0.090909</td>
      <td>0.909091</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>neutral</td>
      <td>cant fall asleep</td>
      <td>-1.0</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>worry</td>
      <td>Choked on her retainers</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>sadness</td>
      <td>Ugh! I have to beat this stupid song to get to...</td>
      <td>-1.0</td>
      <td>0.187500</td>
      <td>0.812500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>sadness</td>
      <td>@BrodyJenner if u watch the hills in london u ...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>surprise</td>
      <td>Got the news</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>sadness</td>
      <td>The storm is here and the electricity is gone</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>love</td>
      <td>@annarosekerr agreed</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>sadness</td>
      <td>So sleepy again and it's not even that late. I...</td>
      <td>-1.0</td>
      <td>0.066667</td>
      <td>0.933333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>worry</td>
      <td>@PerezHilton lady gaga tweeted about not being...</td>
      <td>0.0</td>
      <td>0.058824</td>
      <td>0.882353</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>sadness</td>
      <td>How are YOU convinced that I have always wante...</td>
      <td>-1.0</td>
      <td>0.076923</td>
      <td>0.923077</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
class RemoveUsernameTransformer(TransformerMixin):
    """
    Transformer that will remove tokens from a string of the form:  @someusername
    
    """

    @staticmethod
    def _preprocess_data(data_series):
        """
        inspired from:
        https://raw.githubusercontent.com/youngsoul/ml-twitter-sentiment-analysis/develop/cleanup.py
        :param data_series:
        :return:
        """

        # remove user name
        regex = re.compile(r"@[^\s]+[\s]?")
        data_series.replace(regex, "", inplace=True)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """

        :param X: Series, aka column of data.
        :return:
        """
        RemoveUsernameTransformer._preprocess_data(X)
        return X

```


```python
class RemoveNumbersTransformer(TransformerMixin):
    """
    Transformer that will remove tokens from a string that are numbers.
    """

    @staticmethod
    def _preprocess_data(data_series):
        """
        inspired from:
        https://raw.githubusercontent.com/youngsoul/ml-twitter-sentiment-analysis/develop/cleanup.py
        :param data_series:
        :return:
        """

        # remove numbers
        regex = re.compile(r"\s?[0-9]+\.?[0-9]*")
        data_series.replace(regex, "", inplace=True)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """

        :param X: Series, aka column of data.
        :return:
        """
        RemoveNumbersTransformer._preprocess_data(X)
        return X

```


```python
class StemmedTfidfVectorizer(TfidfVectorizer):
    """
    A TF-IDF Vectorizer that will apply a stemmer to the tokeninze word.
    
    """
    def __init__(self, stemmer=None, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super(StemmedTfidfVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype, norm='l2', use_idf=True, smooth_idf=True,
                             sublinear_tf=False)
        self.stemmer = stemmer

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([self.stemmer.stem(w) for w in analyzer(doc)])

```


```python
def mark_negation_sentence(sentence):
    """
    See the NLTK utility for the mark_negation function.
    
    This function will take a sentence in, split it and call mark_negation, and 
    puts the string back together again.  
    
    Append _NEG suffix to words that appear in the scope between a negation
    and a punctuation mark.
    
    :param sentence an entire sentence
    :return sentence with the negation marked
    """
    return " ".join(mark_negation(sentence.split()))

```


```python
# Functions used to create different features from the text.
def count_username_mentions(value):
    return len(re.findall(r"@[^\s]+[\s]?", value))

def count_ellipsis(value):
    return len(re.findall(r"\.\s?\.\s?\.", value))

def count_hashtags(value):
    return len(re.findall(r"3[^\s]+[\s]?", value))

def count_exclamation_points(value):
    groups = re.findall(r"\w+(!+)", value)
    return sum([len(exclamation_string) for exclamation_string in groups])

def count_question_marks(value):
    groups = re.findall(r"[\w+!](\?+)", value)
    return sum([len(exclamation_string) for exclamation_string in groups])

def is_boredom(y):
    if 'bored' in y.lower() or 'boring' in y.lower():
        return 1
    else:
        return 0

```


```python
def make_features(df):
    df['number_of_mentions'] = df.content.apply(count_username_mentions)
    df['number_of_ellipsis'] = df.content.apply(count_ellipsis)
    df['number_of_exclamations'] = df.content.apply(count_exclamation_points)
    df['number_of_hashtabs'] = df.content.apply(count_hashtags)
    df['number_of_question'] = df.content.apply(count_question_marks)
    df['is_boredom'] = df.content.apply(is_boredom)
    #df['content_len'] = df.content.apply(len)

```


```python
"""
Read the data, setup the function transformers, add the features to the training data.
"""

# -----------------  Function Transformers ----------------
def get_features_df(df):
    return df.loc[:, ['number_of_mentions', 'number_of_ellipsis', 'number_of_exclamations', 'number_of_hashtabs', 'number_of_question', 'is_boredom', 'disposition', 'neg_words', 'neu_words', 'pos_words']]

def get_sentiment_content(df):
    '''Returns the original content from the data set'''
    return df.content.copy()

def get_sentiment_content_negation(df):
    '''Returns the content after it has gone through negation'''
    return df.content_negation.copy()

def get_sentiment_content_preprocess_negation(df):
    '''Returns the content after is has gone through negation AND preprocessed'''
    return df.content_preprocessed_negation


# create a function transformer to just extract the feature columns
get_features_transformer = FunctionTransformer(get_features_df, validate=False)
# usage: get_features_transformer.transform(training_data_with_features).head()

# create a function transformer to return the sentiment content so it can be used in pipeline/union
get_sentiment_content_transformer = FunctionTransformer(get_sentiment_content, validate=False)

get_sentiment_content_negation_transformer = FunctionTransformer(get_sentiment_content_negation, validate=False)

get_sentiment_content_preprocess_negation_transformer = FunctionTransformer(get_sentiment_content_preprocess_negation, validate=False)

# -----------------  End Function Transformers ----------------

def preprocess_data_set(input_data_set):
    # Create a pipeline with the transformers we are keeping, and see the overall improvement.
    # This pipeline gets a little tricky.  the 'get_sentiment_content_transformer' returns a COPY of the
    # original content.  So the transformers work on a copy of the content, leaving the make_features with the
    # original content to create features from.
    preprocessor_pipeline = make_pipeline(get_sentiment_content_transformer, RemoveNumbersTransformer(), RemoveUsernameTransformer())
    preprocessed_training_data_content = preprocessor_pipeline.transform(input_data_set)
    input_data_set['content_preprocessed'] = preprocessed_training_data_content
    input_data_set['content_preprocessed_negation'] = input_data_set.content_preprocessed.apply(mark_negation_sentence)

    make_features(input_data_set)


```


```python
preprocess_data_set(training_data)
```


```python
training_data.shape
```




    (30000, 15)




```python

# create stemmer and vectorizer
stemmer = nltk.stem.SnowballStemmer('english')
stemmed_tfidf_vectorizer = StemmedTfidfVectorizer(stemmer=stemmer, min_df=5, max_df=0.8, ngram_range=(1,4), stop_words='english', sublinear_tf=True)

# UNION
#    PIPE
#        get the negated and preprocessed text
#        send to stemmed tfidf vectorizer to create vocabulary and document term matrix
#    Get the features DataFrame transformer
union = make_union(make_pipeline(get_sentiment_content_preprocess_negation_transformer, stemmed_tfidf_vectorizer),
                  get_features_transformer)

# encode the sentiment outcomes as a number using the LabelEncoder
# would like to create a column, e.g. sentiment_num, which is a numeric representation of the sentiment.
# this will have to also be applied to any test data.
label_encoder = LabelEncoder()

# fit the label encoder with the unique set of sentiments in the training data.
label_encoder.fit(training_data.sentiment.unique())

# Create an outcomes which is the numeric representation of the label
y = training_data.sentiment.apply(lambda x: label_encoder.transform([x])[0])

# create LogisticRegression Model
# 0.3250
model = LogisticRegression(C=0.1)

# 0.307
#model = MultinomialNB()

# Model Pipeline
#    UNION ->Model
# The union will create a traditional vectoizered set of tokens, and a non-DTM data frame for 
# the model.
model_pipeline = make_pipeline(union, model)
cross_val_score(model_pipeline, training_data, y, cv=5, scoring='accuracy').mean()


```




    0.33674193257108043



With the pos/neg/neu/disposition columns, the new accuracy score is now 0.3367


```python
# Read in the test data that will be used to submit to Kaggle
test_data = pd.read_csv('../data/kaggle/sa-emotions/test_data_lexicon.csv')
```


```python
test_data.shape
```




    (10000, 7)




```python
test_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>content</th>
      <th>disposition</th>
      <th>neg_words</th>
      <th>neu_words</th>
      <th>pos_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>is hangin with the love of my life. Tessa McCr...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.916667</td>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>I've Got An Urge To Make Music Like Massively....</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.937500</td>
      <td>0.062500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>@lacrossehawty rofl uh huh</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>@fankri haha! thanks, Tiff   it went well, but...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.962963</td>
      <td>0.037037</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>@alyssaisntcool hahah  i loveeee him though.</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pre-process the test data, because pre-processing should happen on the training and the testing data.
preprocess_data_set(test_data)
test_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>content</th>
      <th>disposition</th>
      <th>neg_words</th>
      <th>neu_words</th>
      <th>pos_words</th>
      <th>content_preprocessed</th>
      <th>content_preprocessed_negation</th>
      <th>number_of_mentions</th>
      <th>number_of_ellipsis</th>
      <th>number_of_exclamations</th>
      <th>number_of_hashtabs</th>
      <th>number_of_question</th>
      <th>is_boredom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>is hangin with the love of my life. Tessa McCr...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.916667</td>
      <td>0.083333</td>
      <td>is hangin with the love of my life. Tessa McCr...</td>
      <td>is hangin with the love of my life. Tessa McCr...</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>I've Got An Urge To Make Music Like Massively....</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.937500</td>
      <td>0.062500</td>
      <td>I've Got An Urge To Make Music Like Massively....</td>
      <td>I've Got An Urge To Make Music Like Massively....</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>@lacrossehawty rofl uh huh</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>rofl uh huh</td>
      <td>rofl uh huh</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>@fankri haha! thanks, Tiff   it went well, but...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.962963</td>
      <td>0.037037</td>
      <td>haha! thanks, Tiff   it went well, but they WO...</td>
      <td>haha! thanks, Tiff it went well, but they WORE...</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>@alyssaisntcool hahah  i loveeee him though.</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>hahah  i loveeee him though.</td>
      <td>hahah i loveeee him though.</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data.shape
```




    (10000, 15)



### Use the model pipeline to make predictions

Once we have the test data preprocessed - we use the model pipeline with all of the training data, and predict on the testing data.

For a pipeline, we can treat it just like a regular model and call 'fit' and 'predict'.



```python
model_pipeline.fit(training_data, y)
```




    Pipeline(memory=None,
         steps=[('featureunion', FeatureUnion(n_jobs=1,
           transformer_list=[('pipeline', Pipeline(memory=None,
         steps=[('functiontransformer', FunctionTransformer(accept_sparse=False,
              func=<function get_sentiment_content_preprocess_negation at 0x10d711620>,
              inv_kw_args=None, inve...ty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False))])




```python
y_pred_class = model_pipeline.predict(test_data)
```


```python
# convert the y_pred_class classification numbers BACK to their string versions for submission
y_pred_class_labels = label_encoder.inverse_transform(y_pred_class)
print(y_pred_class_labels)
```

    ['love' 'neutral' 'neutral' ..., 'happiness' 'happiness' 'neutral']



```python
# create a submission file (resulting score: 0.30040)
# sub1 
# sub2 - added remove html characters
# sub3 = sub1, sanity check
# sub4 = added pos/neg/neu/disposition to the data set.  resulting score: 0.33700
pd.DataFrame({'id':test_data.id, 'sentiment':y_pred_class_labels}).set_index('id').to_csv('../data/kaggle/sa-emotions/sub4.csv')



```
