"""Decoder-preserving sparse autoencoder research utilities."""

from .decoder_distance import (
    batched_ridge_predict,
    calibrate_ridge,
    decoder_distance,
    effective_degrees_of_freedom,
    ridge_hat_matrix,
    ridge_predict,
    sampled_decoder_loss,
)

__all__ = [
    "calibrate_ridge",
    "batched_ridge_predict",
    "decoder_distance",
    "effective_degrees_of_freedom",
    "ridge_hat_matrix",
    "ridge_predict",
    "sampled_decoder_loss",
]
