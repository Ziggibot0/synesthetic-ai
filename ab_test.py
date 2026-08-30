"""A/B probe: does the pipeline learn on CPU (known-good device)?
Same pipeline as voxel_model.run, small config, 2 epochs.
Ground truth for the frozen-loss mystery on the iGPU.
Run: py -3.12 ab_test.py
"""
import os
os.environ["VOXEL_DEVICE"] = "cpu"
import numpy as np
import torch

import voxel_model as V

V.EPOCHS = 2
V.BATCH = 16
V.LAYERS = 2
V.HID = 128
print("AB probe: CPU, 2-layer 128-hid, 2 epochs", flush=True)