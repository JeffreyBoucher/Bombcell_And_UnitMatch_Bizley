# 📦 Import Required Libraries
import numpy as np
import pandas as pd
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

# 📁 Configure Data Paths - Using Processing Playground Dataset

# These are the exact paths from the processing_playground for testing
# calca_302 sessions from 2023-04-19 and 2023-04-20

session_configs = [
    {
        'name': '13052024_AM_Challah_g0',
        'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\13052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        'raw_file': r"D:\F2302_Challah_temp\13052024_AM_Challah_g0\13052024_AM_Challah_g0_imec1\13052024_AM_Challah_g0_t0.imec1.ap.bin",
        'meta_file': r"D:\F2302_Challah_temp\13052024_AM_Challah_g0\13052024_AM_Challah_g0_imec1\13052024_AM_Challah_g0_t0.imec1.ap.meta",
        'save_path': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\13052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output"
    },
    {
        'name': '13052024_PM_Challah_g0',
        'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\13052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        'raw_file': r"D:\F2302_Challah_temp\13052024_PM_Challah_g0\13052024_PM_Challah_g0_imec1\13052024_PM_Challah_g0_t0.imec1.ap.bin",
        'meta_file': r"D:\F2302_Challah_temp\13052024_PM_Challah_g0\13052024_PM_Challah_g0_imec1\13052024_PM_Challah_g0_t0.imec1.ap.meta",
        'save_path':
            r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\13052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output"
    },
    {
        'name': '14052024_AM_Challah_g0',
        'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\14052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        'raw_file': r"D:\F2302_Challah_temp\14052024_AM_Challah_g0\14052024_AM_Challah_g0_imec1\14052024_AM_Challah_g0_t0.imec1.ap.bin",
        'meta_file': r"D:\F2302_Challah_temp\14052024_AM_Challah_g0\14052024_AM_Challah_g0_imec1\14052024_AM_Challah_g0_t0.imec1.ap.meta",
        'save_path'
        r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\14052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output"
    },
    {
        'name': '14052024_PM_Challah_g0',
        'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\14052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        'raw_file': r"D:\F2302_Challah_temp\14052024_PM_Challah_g0\14052024_PM_Challah_g0_imec1\14052024_PM_Challah_g0_t0.imec1.ap.bin",
        'meta_file': r"D:\F2302_Challah_temp\14052024_PM_Challah_g0\14052024_PM_Challah_g0_imec1\14052024_PM_Challah_g0_t0.imec1.ap.meta",
        'save_path': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\14052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output"
    },
    {
        'name': '16052024_AM_Challah_g0',
        'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\16052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        'raw_file': r"D:\F2302_Challah_temp\16052024_AM_Challah_g0\16052024_AM_Challah_g0_imec1\16052024_AM_Challah_g0_t0.imec1.ap.bin",
        'meta_file': r"D:\F2302_Challah_temp\16052024_AM_Challah_g0\16052024_AM_Challah_g0_imec1\16052024_AM_Challah_g0_t0.imec1.ap.meta",
        'save_path':
        # r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\16052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output"
    },
    {
        'name': '16052024_PM_Challah_g0',
        'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\16052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        'raw_file': r"D:\F2302_Challah_temp\16052024_PM_Challah_g0\16052024_PM_Challah_g0_imec1\16052024_PM_Challah_g0_t0.imec1.ap.bin",
        'meta_file': r"D:\F2302_Challah_temp\16052024_PM_Challah_g0\16052024_PM_Challah_g0_imec1\16052024_PM_Challah_g0_t0.imec1.ap.meta",
        'save_path':
        # r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\16052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output"
    },
    {
        'name': '17052024_AM_Challah_g0',
        'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\17052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        'raw_file': r"D:\F2302_Challah_temp\17052024_AM_Challah_g0\17052024_AM_Challah_g0_imec1\17052024_AM_Challah_g0_t0.imec1.ap.bin",
        'meta_file': r"D:\F2302_Challah_temp\17052024_AM_Challah_g0\17052024_AM_Challah_g0_imec1\17052024_AM_Challah_g0_t0.imec1.ap.meta",
        'save_path'
        # r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\17052024_AM_Challah_g0\tempDir\kilosort4_output\sorter_output"
    },
    {
        'name': '17052024_AM_Challah_2_g0',
        'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\17052024_AM_Challah_2_g0\tempDir\kilosort4_output\sorter_output",
        'raw_file': r"D:\F2302_Challah_temp\17052024_AM_Challah_2_g0\17052024_AM_Challah_2_g0_imec1\17052024_AM_Challah_2_g0_t0.imec1.ap.bin",
        'meta_file': r"D:\F2302_Challah_temp\17052024_AM_Challah_2_g0\17052024_AM_Challah_2_g0_imec1\17052024_AM_Challah_2_g0_t0.imec1.ap.meta",
        'save_path'
        # r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\17052024_AM_Challah_2_g0\tempDir\kilosort4_output\sorter_output"
    },

    {
        'name': '17052024_PM_Challah_g0',
        'ks_dir': r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\17052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output",
        'raw_file': r"D:\F2302_Challah_temp\17052024_PM_Challah_g0\17052024_PM_Challah_g0_imec1\17052024_PM_Challah_g0_t0.imec1.ap.bin",
        'meta_file': r"D:\F2302_Challah_temp\17052024_PM_Challah_g0\17052024_PM_Challah_g0_imec1\17052024_PM_Challah_g0_t0.imec1.ap.meta",
        'save_path'
        # r"F:\Louise\Output\tempDir\F2302_Challah\PFC_shank0_Challah\PFC_shank0\everythingAllAtOnce\17052024_PM_Challah_g0\tempDir\kilosort4_output\sorter_output"
    }
]

kilosort_version = 4

print("🎯 BombCell + UnitMatch Pipeline Demo")
print(f"🔬 Sessions to analyze: {len(session_configs)}")
for i, config in enumerate(session_configs):
    print(f"   Session {i + 1}: {config['name']}")
    print(f"      📁 KS: {Path(config['ks_dir']).name}")

print(f"\n🔧 Kilosort version: {kilosort_version}")
print("🎯 This demo will run BombCell first, then use outputs for UnitMatch tracking")


def run_bombcell_session(session_config):
    """Run BombCell on a single session with UnitMatch parameters"""

    name = session_config['name']
    ks_dir = session_config['ks_dir']
    raw_file = session_config.get('raw_file')
    meta_file = session_config.get('meta_file')

    # Create save path in the kilosort directory
    save_path = Path(session_config['save_path'])

    print(f"🔬 Analyzing session: {name}")
    print(f"   📁 Kilosort directory: {ks_dir}")
    print(f"   🗂️  Raw file: {Path(raw_file).name if raw_file else 'Not specified'}")
    print(f"   📄 Meta file: {Path(meta_file).name if meta_file else 'Not specified'}")
    print(f"   💾 Results will be saved to: {save_path}")

    # Check if BombCell has already been run
    existing_results = save_path / "cluster_bc_unitType.tsv"
    if existing_results.exists():
        print(f"   ✅ Found existing BombCell results - loading from disk")

        # Load existing results
        try:
            param, quality_metrics, unit_type_string = bc.load_bc_results(str(save_path))
            unit_type = np.array([1 if ut == 'GOOD' else 0 for ut in unit_type_string])

            print(f"   📊 Loaded results - Total units: {len(unit_type)}")

        except Exception as e:
            print(f"   ⚠️  Error loading existing results: {e}")
            print(f"   🔄 Will re-run BombCell analysis...")
            existing_results = None
    else:
        existing_results = None

    # Run BombCell if no existing results or loading failed
    if not existing_results or not existing_results.exists():
        print("   🚀 Running BombCell analysis...")
        quality_metrics, param, unit_type, unit_type_string = bc.run_bombcell_unit_match(
            ks_dir=ks_dir,
            save_path=str(save_path),
            raw_file=raw_file,
            meta_file=meta_file,
            kilosort_version=kilosort_version
        )

        print(f"   ✅ Analysis complete!")
        print(f"   📊 Total units: {len(unit_type)}")

    # Check UnitMatch waveforms
    raw_waveforms_dir = save_path / "RawWaveforms"
    if raw_waveforms_dir.exists():
        waveform_files = list(raw_waveforms_dir.glob("Unit*_RawSpikes.npy"))
        print(f"   🎯 UnitMatch waveforms: {len(waveform_files)} files saved")
    else:
        print(f"   ⚠️  No UnitMatch waveforms - check raw file access")

    return {
        'name': name,
        'ks_dir': ks_dir,
        'save_path': str(save_path),
        'quality_metrics': quality_metrics,
        'param': param,
        'unit_type': unit_type,
        'unit_type_string': unit_type_string
    }


# Run BombCell on all sessions
print("🚀 Running BombCell analysis on all sessions...")
session_results = []

for i, session_config in enumerate(session_configs):
    print(f"\n{'=' * 60}")
    print(f"SESSION {i + 1}/{len(session_configs)}")
    print(f"{'=' * 60}")
    result = run_bombcell_session(session_config)
    session_results.append(result)

print(f"\n🎉 BombCell analysis complete for {len(session_results)} sessions!")