# import math

import numpy as np

from classification_model.predict import make_prediction

# this function needs to be modified based on use case and
# more test cases can be written depending on use case


def test_make_prediction(sample_input_data):
    # Given
    # expected_first_prediction_value = 113422
    expected_no_predictions = 16281

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    # assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    # assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=100)
