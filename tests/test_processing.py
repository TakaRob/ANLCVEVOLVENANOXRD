from xrd_app.core.processing import run_peaks, run_shapes

def test_pipeline_execution(synthetic_workspace):
    # Pass a real baseline detector and reflection path, but feed it the tiny synthetic data
    result = run_peaks(
        bins_h5=synthetic_workspace["bins_h5"],
        tth_path=synthetic_workspace["tth"],
        detector_path="xrd_app/PeakAlgorithms/5x5_tophat_band_adaptive_snr.py", 
        reflections_path="xrd_app/assets/reflections.py",
        snr_threshold=3.0
    )
    
    assert result["n_bins_with_peaks"] > 0
    assert "0_0" in result["peaks_by_bin"]
    
    # Immediately test shape generation using the peak results
    shape_result = run_shapes(
        peaks=result,
        tth_path=synthetic_workspace["tth"],
        grid_mapping=synthetic_workspace["grid"],
        reflections_path="xrd_app/assets/reflections.py"
    )
    
    # Assert that the framework successfully categorized the outputs
    assert "kept" in shape_result
    assert "filtered" in shape_result
