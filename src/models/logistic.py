from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict


class LogisticRegressionSGD(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: np.ndarray
    bias: float = 0.0

    @classmethod
    def initialize(cls, n_features: int, seed: int = 13) -> Self:
        rng = np.random.default_rng(seed)
        weights = rng.normal(loc=0.0, scale=0.01, size=n_features).astype(np.float32)
        return cls(weights=weights, bias=0.0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights + self.bias
        logits = np.clip(logits, -40, 40)
        return 1.0 / (1.0 + np.exp(-logits))

    def fit_batch(self, x: np.ndarray, y: np.ndarray, learning_rate: float, positive_weight: float = 1.0) -> float:
        probabilities = self.predict_proba(x)
        sample_weights = np.where(y == 1, positive_weight, 1.0).astype(np.float32)
        errors = (probabilities - y) * sample_weights
        gradient_w = x.T @ errors / len(y)
        gradient_b = float(np.mean(errors))
        self.weights -= learning_rate * gradient_w
        self.bias -= learning_rate * gradient_b

        eps = 1e-7
        loss = -np.mean(
            sample_weights * (y * np.log(probabilities + eps) + (1 - y) * np.log(1 - probabilities + eps))
        )
        return float(loss)
