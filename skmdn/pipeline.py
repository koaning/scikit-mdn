"""Sklearn helpers for the MDN."""

from . import MixtureDensityEstimator
from sklearn.pipeline import Pipeline, _final_estimator_has
from sklearn.utils.metaestimators import available_if


class MDNPipeline(Pipeline):
    def __init__(self, steps, **kwargs):
        super().__init__(steps, **kwargs)

    def __getitem__(self, ind):
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")

            return Pipeline(self.steps[ind], memory=self.memory, verbose=self.verbose)

        try:
            name, est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est

    def fit(self, X, y, **kwargs):
        if not isinstance(self.steps[-1][1], MixtureDensityEstimator):
            raise ValueError("Last step must be a MixtureDensityEstimator.")

        return super().fit(X, y, **kwargs)

    def transform(self, X):
        return self[:-1].transform(X)

    def inverse_transform(self, X):
        return self[:-1].inverse_transform(X)

    @available_if(_final_estimator_has("pdf"))
    def pdf(self, X, **kwargs):
        X_ = self.transform(X)
        return self._final_estimator.pdf(X_, **kwargs)

    @available_if(_final_estimator_has("cdf"))
    def cdf(self, X, **kwargs):
        X_ = self.transform(X)
        return self._final_estimator.cdf(X_, **kwargs)

    @available_if(_final_estimator_has("forward"))
    def forward(self, X, **kwargs):
        X_ = self.transform(X)
        return self._final_estimator.forward(X_, **kwargs)
