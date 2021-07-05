from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, _VectorizerMixin
import numpy as np


class BasketVectorizer:
    """
    This class is responsible for converting a basket into one hot vectors
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self._cv = CountVectorizer(vocabulary=self.vocabulary, token_pattern=r"\b\d+\b", preprocessor=self._preprocess)

    @staticmethod
    def _preprocess(doc):
        """
        The default vectorizer only accepts list of strings, we also want it to accept list of list or list of arrays
        """
        if isinstance(doc, list) or isinstance(doc, np.ndarray):
            return ",".join(doc)
        return doc

    def transform(self, X, toarray=False):
        if toarray:
            return self._cv.transform(X).toarray()
        else:
            return self._cv.transform(X)

    def get_cv(self):
        return self._cv


if __name__ == '__main__':
    cab = {str(i) for i in range(1, 11)}
    bc = BasketVectorizer(vocabulary=cab)
    print(bc.get_cv().get_feature_names())
    res = bc.transform([["1", "2"], "7,8,9,10"], toarray=True)
    print(res)
