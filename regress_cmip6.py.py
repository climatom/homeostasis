#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute pattern-scaling coefficients
"""

from numba import njit, prange
import numpy as np
import xarray as xa
import os

# Put numba regression in here

# GTAS files
tasdir="/home/lunet/gytm3/Homeostasis/CMIP6/GTAS/"
tasfiles=[ii for ii in os.listdir(tasdir) if ".nc" in ii]

# MDI files
moddir="/home/lunet/gytm3/Homeostasis/CMIP6/Merged/Regridded/"
mdi_files=[ii for ii in os.listdir(moddir) if ".mdi." in ii]

# TW files
tw_files=[ii for ii in os.listdir(moddir) if ".tw." in ii]







