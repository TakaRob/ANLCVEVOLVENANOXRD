import numpy as np
from xrd_app.core.io import build_regular_grid
from xrd_app.ShapeAlgorithms.gaussian import link_peaks

def test_build_regular_grid():
    # Test uniform raster synthesis
    n_total = 100
    n_cols = 10
    row, col, n_rows, final_cols = build_regular_grid(n_total, n_cols)
    
    assert n_rows == 10
    assert final_cols == 10
    # Check boustrophedon (serpentine) behavior on row 1
    assert col[10] == 9 
    assert col[19] == 0

def test_spatial_linking():
    # Provide dummy detections simulating 3 neighboring bins
    mock_detections = {
        "0_0": [{"x": 50, "y": 50, "snr": 10, "label": "Perovskite"}],
        "0_1": [{"x": 51, "y": 49, "snr": 8,  "label": "Perovskite"}], # Close enough to link
        "1_0": [{"x": 200, "y": 200, "snr": 5, "label": "PbI2"}],      # Far away, different feature
    }
    
    # link_tolerance defaults to 5 pixels
    features = link_peaks(mock_detections, n_rows=2, n_cols=2, link_tolerance=5)
    
    assert len(features) == 2 # Should group into 2 distinct features
    
    # The perovskite feature should contain 2 bins
    perovskite_feature = next(f for f in features if len(f) == 2)
    assert perovskite_feature[0][0] in ["0_0", "0_1"]
