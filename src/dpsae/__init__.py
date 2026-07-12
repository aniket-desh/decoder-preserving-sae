"""Decoder-preserving sparse autoencoder research utilities."""

from .decoder_distance import decoder_distance, ridge_hat_matrix

__all__ = ["decoder_distance", "ridge_hat_matrix"]

