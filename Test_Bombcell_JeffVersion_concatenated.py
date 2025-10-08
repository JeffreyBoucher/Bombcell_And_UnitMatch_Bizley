######### FIRST CELL ###########


# 📦 Import Required Libraries
import numpy as np
import pandas as pd
import os as os_true
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import BombCell for quality control
import bombcell as bc

# Import UnitMatch for cross-session tracking (optional - only if running Part 2)
try:
    import UnitMatchPy.bayes_functions as bf
    import UnitMatchPy.utils as util
    import UnitMatchPy.overlord as ov
    import UnitMatchPy.save_utils as su
    import UnitMatchPy.GUI as um_gui
    import UnitMatchPy.assign_unique_id as aid
    import UnitMatchPy.default_params as default_params
    UNITMATCH_AVAILABLE = True
    print("✅ BombCell and UnitMatch imported successfully")
except ImportError as e:
    UNITMATCH_AVAILABLE = False
    print("✅ BombCell imported successfully")
    print("⚠️  UnitMatch not available - please install: pip install UnitMatchPy")
    print(f"    Error: {e}")

print("🚀 Ready to analyze neural data!")



######### SECOND CELL ###########
session_configs = [
        {
            'name': 'weekOf20240513_PFC_F2302_Challah', ### just made this up
            'ks_dir': r"F:\Jeff\Ferrets_Spikesorted\F2302_Challah\spikesorted\tempDir\PFC_shank0_Challah\PFC_shank0\weekOf20240513\tempDir\kilosort4_output\sorter_output", ### copy location
            'raw_file': r"D:\F2302_Challah_temp\13052024_AM_Challah_g0\13052024_AM_Challah_g0_imec1\13052024_AM_Challah_g0_t0.imec1.ap.bin", ### don't have yet, will be on an external drive
            'meta_file': r"None", ### might never exist. Might have to make a fake one.
            'save_path': r"F:\Jeffrey\BombcellTestOutput" ### Louise set this as the same as ks_dir... The way I am doing it I might make a clone.
        },
        # {
        #     'name': '13052024_PM_Challah_g0',
        #     'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\13052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        #     'raw_file': r"D:\F2302_Challah_temp\13052024_PM_Challah_g0\13052024_PM_Challah_g0_imec1\13052024_PM_Challah_g0_t0.imec1.ap.bin",
        #     'meta_file': r"D:\F2302_Challah_temp\13052024_PM_Challah_g0\13052024_PM_Challah_g0_imec1\13052024_PM_Challah_g0_t0.imec1.ap.meta",
        #     'save_path': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\13052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output"
        # },
        # {
        #     'name': '14052024_AM_Challah_g0',
        #     'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\14052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        #     'raw_file': r"D:\F2302_Challah_temp\14052024_AM_Challah_g0\14052024_AM_Challah_g0_imec1\14052024_AM_Challah_g0_t0.imec1.ap.bin",
        #     'meta_file': r"D:\F2302_Challah_temp\14052024_AM_Challah_g0\14052024_AM_Challah_g0_imec1\14052024_AM_Challah_g0_t0.imec1.ap.meta",
        #     'save_path': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\14052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output"
        # }#,
    ]
kilosort_version = 4

print("🎯 BombCell + UnitMatch Pipeline Demo")
print("📊 Dataset: calca_302 cross-session tracking")
print(f"🔬 Sessions to analyze: {len(session_configs)}")
for i, config in enumerate(session_configs):
    print(f"   Session {i + 1}: {config['name']}")
    print(f"      📁 KS: {Path(config['ks_dir']).name}")

print(f"\n🔧 Kilosort version: {kilosort_version}")
print("🎯 This demo will run BombCell first, then use outputs for UnitMatch tracking")

######### THIRD CELL ###########