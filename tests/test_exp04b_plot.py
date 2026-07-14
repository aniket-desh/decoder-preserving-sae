from experiments.plot_exp04b import selected_row, sparsity_from_name, style_method


def test_plot_method_mapping_and_frozen_row_are_stable():
    assert style_method("dpsae") == "isotropic"
    assert style_method("spectral") == "spectral"
    assert selected_row([{"features": 4}, {"features": 8}], 8) == {
        "features": 8
    }
    assert sparsity_from_name("dpsae_k16_s0") == 16
    assert sparsity_from_name("dpsae_s0") == 32
    assert sparsity_from_name("dpsae_k64_s0") == 64
