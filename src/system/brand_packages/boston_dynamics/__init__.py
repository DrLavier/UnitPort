#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boston Dynamics brand package."""

from .spot_model import SpotModel

SUPPORTED_MODELS = ["spot"]

__all__ = ["SpotModel", "SUPPORTED_MODELS"]
