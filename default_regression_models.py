#!/usr/bin/env python3
"""Inference-time compatibility module for legacy PLifePred2 model pickles.

The structure model was serialized with a custom transformer whose import path
was ``default_regression_models.DescriptorPreprocessor``.  Joblib needs that
same module path when loading the estimator.  This file intentionally contains
only the transformer definition required for deserialization; it does NOT
contain the model-training pipeline.

Keep the class name, module name, constructor signature, and implementation
synchronized with the original training-time definition.  Do not rename this
module unless the structure model is re-serialized with a new import path.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FLOAT32_MAX = float(np.finfo(np.float32).max)
DEFAULT_EXCLUDE = ("Name", "SMILES", "label", "log_label")


class DescriptorPreprocessor(BaseEstimator, TransformerMixin):
    """Training-fitted descriptor cleaning and missing-value preprocessing."""

    def __init__(self, exclude_cols=DEFAULT_EXCLUDE):
        self.exclude_cols = exclude_cols

    @staticmethod
    def _to_numeric(df):
        return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        excl = set(self.exclude_cols or [])
        self.input_columns_ = [c for c in X.columns if c not in excl]
        Xn = self._to_numeric(X[self.input_columns_])

        kept, indicator_src, medians, audit = [], [], {}, []
        n = len(Xn)
        for col in self.input_columns_:
            s = Xn[col]
            nonmissing = s.dropna()
            n_missing = int(s.isna().sum())
            nunique = int(nonmissing.nunique())
            max_abs = (
                float(np.abs(nonmissing.to_numpy(dtype=float)).max())
                if len(nonmissing) else np.nan
            )

            if n_missing == n:
                action, reason = "DROP", "all_nan_in_training"
            elif nunique <= 1:
                action, reason = "DROP", "constant_or_zero_variance_in_training"
            elif max_abs > FLOAT32_MAX:
                action, reason = "DROP", "exceeds_float32_range_in_training"
            else:
                medians[col] = float(nonmissing.median())
                kept.append(col)
                if n_missing > 0:
                    indicator_src.append(col)
                    action, reason = "KEEP_FILL_MEDIAN_ADD_INDICATOR", "partial_missingness"
                else:
                    action, reason = "KEEP", "valid"

            audit.append({
                "feature": col,
                "action": action,
                "reason": reason,
                "train_missing_n": n_missing,
                "train_missing_fraction": n_missing / n,
                "training_median_used": medians.get(col, np.nan),
                "train_unique_nonmissing": nunique,
                "train_max_abs": max_abs,
            })

        self.kept_features_ = kept
        self.indicator_sources_ = indicator_src
        self.medians_ = pd.Series(medians, dtype=float)
        self.feature_names_out_ = kept + [f"{c}__missing" for c in indicator_src]
        self.audit_ = pd.DataFrame(audit)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        absent = [c for c in self.input_columns_ if c not in X.columns]
        if absent:
            raise ValueError(
                f"Input is missing {len(absent)} required descriptor columns. "
                f"Examples: {absent[:10]}"
            )
        Xn = self._to_numeric(X[self.input_columns_])
        kept = Xn[self.kept_features_]
        indicators = kept[self.indicator_sources_].isna().astype(np.int8)
        filled = kept.fillna(self.medians_)
        out = np.concatenate(
            [filled.to_numpy(dtype=np.float64), indicators.to_numpy(dtype=np.float64)],
            axis=1,
        )
        return np.clip(out, -FLOAT32_MAX, FLOAT32_MAX)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

