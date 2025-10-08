# 📦 Import Required Libraries
import numpy as np
import pandas as pd
import os as os_true
import pickle
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

###### some cherrypicked functions I am importing manually
def sort_np_sessions(
        sessions_list,
        minimum_duration_s=-1,
):
    """
    Sorts a list of Neurophysiology (NP) session directories based on the file creation time of their metadata files.

    Parameters:
    -----------
    sessions_list : list of pathlib.Path objects or list of str
        A list of pathlib.Path objects or a list of strings representing the directories of NP sessions.
    minimum_duration_s : int, optional
        The minimum duration (in seconds) of sessions to be included in the sorted list. Defaults to -1, which includes all sessions.

    Returns:
    --------
    numpy.ndarray
        A 1-dimensional array of pathlib.Path objects representing the directories of NP sessions, sorted in ascending order
        of their file creation times.
    """

    if isinstance(sessions_list[0], str):
        sessions_list = [Path(s) for s in sessions_list]

    meta_dicts = []
    for session in sessions_list:
        metafile = [f for f in session.glob('*.meta')][0]
        meta = load_meta_file(metafile)
        meta['session_name'] = session
        meta_dicts.append(meta)

    df_meta = pd.DataFrame.from_dict(meta_dicts)
    df_meta['fileCreateTime'] = pd.to_datetime(df_meta['fileCreateTime'])
    df_meta = df_meta.sort_values('fileCreateTime', ignore_index=True)

    df_meta = df_meta.loc[df_meta.fileTimeSecs > minimum_duration_s]

    return df_meta.session_name.to_numpy()

def load_meta_file(metafile):
    ''' Read a .meta file from spikeglx and return a dictionnary of its content'''
    # load metafile
    meta = {}
    with open(metafile, 'r') as f:
        for ln in f.readlines():
            tmp = ln.split('=')
            k, val = tmp[0], ''.join(tmp[1:])
            k = k.strip()
            val = val.strip('\r\n')
            if '~' in k:
                meta[k] = val.strip('(').strip(')').split(')(')
            else:
                try:  # is it numeric?
                    meta[k] = float(val)
                except:
                    meta[k] = val
    return meta

###### SECOND CELL ########

if True: # iterating way to get session configs for a range of sessions. Uses a copy-paste of the strategy used in spikesorting.
    ###PARAMETERS
    session_path = Path('Z:/Data/Neuropixels/F2302_Challah/')  # path to where all relevant sessions are stored
    ferret = 'F2302_Challah'
    recordingZone = 'PFC_shank0_Challah'

    if recordingZone == 'PFC_shank0_Challah':
        stream_id = 'imec1.ap'
        # sessionSetLabel = 'TwentiesOfMay'
        # sessionSetLabel = "AllJune"
        sessionSetLabel = "AllMay"
        # sessionSetLabel = 'Tens_Of_June'
        channel_map_to_use = 'Challah_top_PFC_shank0.imro'
        badChannelList = [21, 109, 133, 170, 181, 202, 295, 305, 308, 310, 327, 329, 339]
        # something else also. Need to read metadata
        manuallySplitShank = False
        if manuallySplitShank:
            manualShankSplitRanged = [[0,3625],[3625,4500],[4500,10000]]
            manualShankSplitNames = ["Top","Middle","Bottom"]
        else:
            manualShankSplitRanged = [[0, 10000]]
            manualShankSplitNames = ["All"]
    elif recordingZone == 'PFC_shank3_Challah':
        stream_id = 'imec1.ap'
        # sessionSetLabel = 'TwentiesOfMay'
        sessionSetLabel = "AllJune"
        # sessionSetLabel = 'Tens_Of_June'
        channel_map_to_use = 'Challah_top_PFC_shank3.imro'
        badChannelList = [21, 109, 133, 170, 181, 202, 295, 305, 308, 310, 327, 329, 339]
        # something else also. Need to read metadata
        manuallySplitShank = False
        if manuallySplitShank:
            manualShankSplitRanged = [[0,3625],[3625,4500],[4500,10000]]
            manualShankSplitNames = ["Top","Middle","Bottom"]
        else:
            manualShankSplitRanged = [[0, 10000]]
            manualShankSplitNames = ["All"]
    # output_folder = Path('F:/Louise/Output')
    output_folder = Path('F:/Jeff/Output')

    if sessionSetLabel == 'All_ACx_Top':
        sessionString = '[0-9][0-9]*'  ### this actually selects more than just the top
    elif sessionSetLabel == 'Tens_Of_June':
        sessionString = '1[0-9]06*'
    elif sessionSetLabel == 'TheFirstDay':
        sessionString = '1305*'
    elif sessionSetLabel == 'TheFirstSession':
        sessionString = '1305*AM*'
    elif sessionSetLabel == "TwentiesOfMay":
        sessionString = '2[0-9]052024*'
    elif sessionSetLabel == 'TensOfMay':
        sessionString = '1[0-9]052024*'
    elif sessionSetLabel == "AllJuly":
        sessionString = '*072024*'
    elif sessionSetLabel == "AllJune":
        sessionString = '*062024*'
    elif sessionSetLabel == "AllMay":
        sessionString = '*052024*'
    SessionsInOrder = sort_np_sessions(list(session_path.glob(sessionString)))
    sessionSetName = sessionSetLabel

    for i, session in enumerate(SessionsInOrder):
        temp_session_configs = [{'name': [],
                            'ks_dir': [],
                            'raw_file': [],
                            'meta_file': [],
                            'save_path': []}]

        if not i:
            session_configs = temp_session_configs
            custom_bombcell_paths = []
            custom_raw_waveform_paths  = []
        else:
            session_configs = session_configs + temp_session_configs
        session_name = session.name
        session_configs[i]["name"] = session_name
        working_dir = output_folder / 'tempDir' / ferret / session_name
        dp = session_path / session_name

        probeFolder = list(dp.glob('*' + stream_id[:-3]))
        probeFolder = probeFolder[0]  # name of probe folder
        session_configs[i]["meta_file"] = probeFolder / (session_name + '_t0.' + stream_id + '.meta')

        ### hey by the way, apparently they prefer if your raw data is phase corrected and CAR. So I should maybe save some raws with SI that meet that requirement...
        session_configs[i]["raw_file"] = list(probeFolder.glob('*.bin'))[0] #obviously only works for bins, not cbin
        session_configs[i]["ks_dir"] = output_folder / 'tempDir' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name /  Path("tempDir/kilosort4_output/sorter_output/")
        session_configs[i]["save_path"] = output_folder / 'tempDir' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name /  Path("tempDir/kilosort4_output/sorter_output/")
        custom_bombcell_paths  = custom_bombcell_paths  + [session_configs[i]["ks_dir"] / Path("cluster_bc_unitType.tsv")]
        custom_raw_waveform_paths = custom_raw_waveform_paths + [session_configs[i]["ks_dir"] / Path("RawWaveforms")]
        breakpointGoesHere = True

spike_positions = np.load(ks_dir / "spike_positions.npy")
print('br')
