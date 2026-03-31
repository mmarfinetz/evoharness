"""Editable candidate for the command_repo demo."""

BIAS = 1.0
LINEAR = 0.0
QUADRATIC = 0.25


def predict(x: float) -> float:
    return float(BIAS + (LINEAR * x) + (QUADRATIC * x * x))
