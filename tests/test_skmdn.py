import pytest 
from skmdn import MixtureDensityEstimator

@pytest.mark.parametrize(
    "y_bound,y_bound_,y_bound_correct",
    [
        (0, 10, 0),
        (10, 0, 10),

        (None, 0, 0),
        (None, 10, 10),
    ]
)
def test_validate_y_bound(y_bound, y_bound_, y_bound_correct):
    mdn = MixtureDensityEstimator()
    assert mdn.validate_y_bound(y_bound, y_bound_) == y_bound_correct