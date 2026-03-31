from __future__ import annotations

import json
import math

from candidate import predict


TARGET_POINTS = (
    (-2.0, 5.75),
    (-1.0, 3.0),
    (0.0, 1.75),
    (1.0, 2.0),
    (2.0, 3.75),
)


def compute_metrics() -> dict[str, float]:
    errors = [predict(x_value) - expected for x_value, expected in TARGET_POINTS]
    squared_error = sum(error * error for error in errors) / len(errors)
    rmse = math.sqrt(squared_error)
    max_abs_error = max(abs(error) for error in errors)
    score = 1.0 / (1.0 + rmse)
    return {
        "score": score,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
    }


if __name__ == "__main__":
    print(json.dumps(compute_metrics(), sort_keys=True))
