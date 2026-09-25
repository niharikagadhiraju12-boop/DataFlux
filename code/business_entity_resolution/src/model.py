"""
Matching Model Module for Business Entity Resolution.

Responsibilities:
- Training and configuring the classification model
- Generating match probabilities
- Generating optional threshold-based predictions
- Saving and loading trained model artifacts
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


class EntityMatchingModel:
    """
    Wrapper for a binary classification model used for
    business entity matching.
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the matching model.

        Parameters
        ----------
        model_params:
            Optional Logistic Regression hyperparameters.
        """

        default_params = {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        }

        if model_params:
            default_params.update(model_params)

        self.model_params = default_params
        self.model: Optional[LogisticRegression] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> "EntityMatchingModel":
        """
        Train the entity matching classifier.
        """

        self.model = LogisticRegression(
            **self.model_params
        )

        self.model.fit(X, y, **kwargs)

        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> Any:
        """
        Generate probability that each candidate pair is a match.
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted yet."
            )

        return self.model.predict_proba(X)

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Any:
        """
        Generate binary match predictions.

        Threshold selection is intentionally kept outside
        model training so that validation can tune it
        for the target metric.
        """

        probabilities = self.predict_proba(X)[:, 1]

        return (probabilities >= threshold).astype(int)

    def save(
        self,
        filepath: Union[str, Path],
    ) -> None:
        """
        Save the trained model to disk.
        """

        if self.model is None:
            raise RuntimeError(
                "Cannot save an unfitted model."
            )

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model": self.model,
                "model_params": self.model_params,
            },
            filepath,
        )

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
    ) -> "EntityMatchingModel":
        """
        Load a previously saved model.
        """

        filepath = Path(filepath)

        artifact = joblib.load(filepath)

        instance = cls(
            model_params=artifact["model_params"]
        )

        instance.model = artifact["model"]

        return instance