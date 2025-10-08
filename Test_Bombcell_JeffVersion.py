### this will probably be exactly the same as "Test_Bombcell_UnitMatch.py" but I don't want to overwrite Louise's file.

############ FIRST CELL ##############


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


kilosort_version = 4

print("🎯 BombCell + UnitMatch Pipeline Demo")
print("📊 Dataset: calca_302 cross-session tracking")
print(f"🔬 Sessions to analyze: {len(session_configs)}")
for i, config in enumerate(session_configs):
    print(f"   Session {i + 1}: {config['name']}")
    print(f"      📁 KS: {Path(config['ks_dir']).name}")

print(f"\n🔧 Kilosort version: {kilosort_version}")
print("🎯 This demo will run BombCell first, then use outputs for UnitMatch tracking")



####### THIRD CELL ############


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
    reloadExistingResults = True
    if existing_results.exists()&reloadExistingResults:
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


###### FORTH CELL ########

def extract_motion(ks_dir, file_name):
    motion = []
    space_bins = []

    df = pd.read_csv(ks_dir / file_name)
    #sessions = df['session']
    # df_list = df.values.tolist()
    # for i in sessions:
    #     motion.append(df['motion'][df['session'] == i])
    #     space_bins.append(df['space'][df['session'] == i])

    return df #motion, space_bins, sessions

def intersection(A,B):
    U = []
    for i in A:
        if i in B:
            U.append(i)
    return U


def correct_motion_on_channel_pos(positions, motion_week, motion_sessions, session, week_session_correspondance):
    #getting id of this session's week + of all the other sessions of this week
    find_week= np.where(week_session_correspondance[:,1]==session)
    week= week_session_correspondance[find_week[0], 0][0]
    find_sessions= np.where(week_session_correspondance[:,0]==week)
    week_sessions= week_session_correspondance[find_sessions[0], 1][0]

    x_positions = positions[:,0].copy()
    y_positions_init = positions[:,1].copy()

    if week==0 and session== 0:
        return positions
    if week==0 and not(session==0):
        for i in range(week_sessions,session):
            new_positions = []
            motion_df = motion_sessions[motion_sessions['session'] ==i]
            for j,pos in enumerate(y_positions_init):
                bins_center= np.array(motion_df['center_space_bin'])
                bin_low= list(np.where(pos <= bins_center+25)) #we look for the indice of the
                bin_high = list(np.where(pos >= bins_center-25))
                if len(bin_low[0])!=0 and len(bin_high[0])!=0:
                   bin_loc= intersection(bin_low[0],bin_high[0])[0]
                else:
                    if len(bin_low[0])==0:
                        bin_loc = bin_high[0][-1]
                    elif len(bin_high[0])==0:
                        bin_loc = bin_low[0][0]
                local_motion = motion_df['motion'].iloc[bin_loc]
                new_positions.append(pos - local_motion)
            y_positions_init= new_positions.copy()
        all_positions = np.array([list(x_positions), list(new_positions)])
        all_positions = all_positions.T
        return all_positions
    else:
        for i in range(week):
            new_positions = []
            motion_df = motion_week[motion_week['session_week'] ==i]
            for j,pos in enumerate(y_positions_init):
                bins_center= np.array(motion_df['center_space_bin'])
                bin_low= list(np.where(pos <= bins_center+25)) #we look for the indice of the
                bin_high = list(np.where(pos >= bins_center-25))
                if len(bin_low[0])!=0 and len(bin_high[0])!=0:
                   bin_loc= intersection(bin_low[0],bin_high[0])[0]
                else:
                    if len(bin_low[0])==0:
                        bin_loc = bin_high[0][-1]
                    elif len(bin_high[0])==0:
                        bin_loc = bin_low[0][0]
                local_motion = motion_df['motion'].iloc[bin_loc]
                new_positions.append(pos - local_motion)
            y_positions_init= new_positions.copy()
        for i in range(week_sessions,session):
            new_positions = []
            motion_df = motion_sessions[motion_sessions['session'] ==i]
            for j,pos in enumerate(y_positions_init):
                bins_center= np.array(motion_df['center_space_bin'])
                bin_low= list(np.where(pos <= bins_center+25)) #we look for the indice of the
                bin_high = list(np.where(pos >= bins_center-25))
                if len(bin_low[0])!=0 and len(bin_high[0])!=0:
                    if len(intersection(bin_low[0],bin_high[0])) == 1: #when computing new bins centers for weeks, distance between bins changes
                        bin_loc= intersection(bin_low[0],bin_high[0])[0]
                    else:
                        dist_low= abs(pos-bins_center[bin_low[0][0]])
                        dist_high= abs(bins_center[bin_high[0][-1]]-pos)
                        if dist_low>dist_high:
                            bin_loc= bin_high[0][0]
                        else:
                            bin_loc= bin_low[0][-1]
                else:
                    if len(bin_low[0])==0:
                        bin_loc = bin_high[0][-1]
                    elif len(bin_high[0])==0:
                        bin_loc = bin_low[0][0]
                local_motion = motion_df['motion'].iloc[bin_loc]
                new_positions.append(pos - local_motion)
            y_positions_init= new_positions.copy()
        print(f'{j}')
        all_positions = np.array([list(x_positions), list(new_positions)])
        all_positions = all_positions.T
        return all_positions


print("🎯 Setting up UnitMatch for cross-session tracking...")

# Get default UnitMatch parameters
um_param = default_params.get_default_param()

# Set up paths from our BombCell results
KS_dirs = [result['ks_dir'] for result in session_results]
um_param['KS_dirs'] = KS_dirs

#### I set up the "custom_bombcell_paths" in cell 2 now.

spike_positions = np.array([]) np.load(ks_dir / "spike_positions.npy")
spike_clusters = np.load(ks_dir / "spike_clusters.npy")
spike_times = np.load(ks_dir / "spike_times.npy")


print(f"📁 Kilosort directories: {len(KS_dirs)}")
for i, ks_dir in enumerate(KS_dirs):
    print(f"   Session {i + 1}: {Path(ks_dir).name}")
    spike_positions= np.concatenate(spike_positions,np.load(ks_dir / "spike_positions.npy"))
    spike_clusters = np.concatenate(spike_clusters,np.load(ks_dir / "spike_clusters.npy"))
    spike_times = np.concatenate(spike_times,np.load(ks_dir / "spike_times.npy"))

print(f"📊 BombCell unit classifications:")
for i, bc_path in enumerate(custom_bombcell_paths):
    exists = "✅" if Path(bc_path).exists() else "❌"
    print(f"   Session {i + 1}: {exists} {Path(bc_path).name}")

print(f"🎯 Raw waveforms for UnitMatch:")
for i, wv_path in enumerate(custom_raw_waveform_paths):
    exists = "✅" if Path(wv_path).exists() else "❌"
    n_files = len(list(Path(wv_path).glob("Unit*_RawSpikes.npy"))) if Path(wv_path).exists() else 0
    print(f"   Session {i + 1}: {exists} {n_files} waveform files")

motion_week= extract_motion(ks_dir, 'motion_weeks.csv')
motion_sessions= extract_motion(ks_dir, 'motion_sessions.csv')
week_session_correspondance = np.load(ks_dir / "session_to_week_id.npy")


# Setup UnitMatch paths - this matches exactly what processing_playground does
try:
    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(  #
        KS_dirs,
        custom_raw_waveform_paths=custom_raw_waveform_paths,
        custom_bombcell_paths=custom_bombcell_paths,
        motion_week=motion_week,
        motion_sessions=motion_sessions,
        week_session_correspondance=week_session_correspondance
    )

    #motion, space_bins, sessions = extract_motion(ks_dir

    um_param = util.get_probe_geometry(channel_pos[0], um_param) ### if I was to manually create virtual shanks, this might be where to do it...
    print("✅ UnitMatch paths configured successfully")
    UNITMATCH_READY = True

except Exception as e:
    print(f"❌ Error setting up UnitMatch paths: {e}")
    print("   Make sure BombCell has been run and waveforms extracted")
    UNITMATCH_READY = False

########## FIFTH CELL #############

# %%
# 🧠 Run UnitMatch Analysis Pipeline


print("🚀 Running UnitMatch cross-session analysis...")
if manuallySplitShank:
    for (iiiii,c_pos) in enumerate(channel_pos):
        if not iiiii:
            continue
        if not (channel_pos[0] == c_pos).all():
            print("warning, you don't seem to have all c_pos equal. This shouldn't happen unless you mixed channel maps. Still assuming c_pos[0] is representative")
    shankSplitSubsets = np.full((np.shape(channel_pos[0])[0],len(manualShankSplitRanged)), False)
    for (iiiii,shankRange) in enumerate(manualShankSplitRanged):
        shankSplitSubsets[:,iiiii] = ((shankRange[0] <= channel_pos[0][:,2])&(shankRange[1] > channel_pos[0][:,2]))
else:
    shankSplitSubsets = np.full((np.shape(channel_pos[0])[0],1), True)
# Step 0: Load good units and waveform data
print("📊 Step 0: Loading unit waveforms...")
for (iiiii,shankSplitSubset) in enumerate(shankSplitSubsets.T):
    shankSplitSubset =shankSplitSubsets[:,iiiii] # later I will loop
    waveform, session_id, session_switch, within_session, good_units, um_param = util.load_good_waveforms(
        wave_paths, unit_label_paths, um_param, good_units_only=True,splitShankActive=True,channel_pos=channel_pos,shankSplitSubset=shankSplitSubset ### good units only actually seems to include MUA, I think? Should double check, but following it with breakpoints, that seems to be the case. I might still want this false because tracking noise might be of interest... One expects noise to not track across sessions, and also some noise should really be MUA...
    )

    print(f"   ✅ Loaded {len(np.concatenate(good_units))} good units across {len(session_results)} sessions")
    print(f"   📏 Waveform shape: {waveform.shape} (units × time × channels)")

    # Create cluster info
    clus_info = {
        'good_units': good_units,  # modified to Good and MUA
        'session_switch': session_switch,
        'session_id': session_id,
        'original_ids': np.concatenate(good_units)
    }

    spike_positions = np.load(ks_dir / "spike_positions.npy")
    spike_clusters = np.load(ks_dir / "spike_clusters.npy")
    spike_times = np.load(ks_dir / "spike_times.npy")

    df_spikes= pd.DataFrame(list(zip(spike_positions, spike_clusters, spike_times)), columns=["Position", "Clus", "Times"])
    df_spikes.sort_values("Clus", ascending=True, inplace=True)

    spike_positions = np.array(df_spikes["Position"])
    spike_clusters = np.array(df_spikes["Clus"])
    spike_times = np.array(df_spikes["Times"])

    spike_pos_by_clus= []
    clus=spike_clusters[0]
    spike_for_this_clus = []
    count=0
    for i, pos in enumerate(spike_positions):
        if spike_clusters[i] == clus:
            spike_for_this_clus.append(pos)
        else:
            spike_pos_by_clus.append(spike_for_this_clus.copy())
            spike_for_this_clus=[]
            spike_for_this_clus.append(pos)
            clus= spike_clusters[i]
    spike_pos_by_clus.append(spike_for_this_clus)

    session_clus= []
    session_here= 0
    for i in um_param['n_units_per_session']:
        for j in range(i):
            session_clus.append(session_here)
        session_here+=1
    full_number=0
    for k in spike_pos_by_clus:
        full_number+= len(k)
    print(full_number)

    ##Test spike loc
    motion_corrected_pos=[]
    for k,positions in enumerate(spike_pos_by_clus):
        positions_this_clus = np.vstack(positions)
        motion_corrected_clus= correct_motion_on_channel_pos(positions_this_clus, motion_week, motion_sessions, session_clus[k], week_session_correspondance)
        motion_corrected_pos.append(motion_corrected_clus)

    motion_corrected_all_spikes = []
    for k in motion_corrected_pos:
        for j in k:
            motion_corrected_all_spikes.append(j)
    print('pause and check')

    y_pos_init= [spike_positions[i][1] for i in range(len(spike_positions))]
    y_pos_corrected= [motion_corrected_all_spikes[i][1] for i in range(len(motion_corrected_all_spikes))]

    fig, ax = plt.subplots()
    ax.scatter( spike_times, y_pos_init , s=5)#marker="o", color="blue", markersize= 5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter( spike_times, y_pos_corrected , s=5)# marker="o", color="blue", markersize= 5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    plt.show()


    # Step 1: Extract waveform parameters
    print("🔍 Step 1: Extracting waveform features...")
    extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, um_param)

    print("   ✅ Extracted amplitude, spatial decay, and waveform features")
    # Steps 2-4: Calculate similarity metrics with drift correction
    print("📐 Steps 2-4: Computing similarity metrics and drift correction...")
    total_score, candidate_pairs, scores_to_include, predictors = ov.extract_metric_scores(
        extracted_wave_properties, session_switch, within_session, um_param, niter=2
    )
    print(f"   ✅ Found {np.sum(candidate_pairs)} candidate unit pairs")
    print(f"   📊 Metrics included: {list(scores_to_include.keys())}")

    # Step 5: Probability analysis using Naive Bayes
    print("🧮 Step 5: Calculating match probabilities...")

    # Set up priors
    prior_match = 1 - (um_param['n_expected_matches'] / um_param['n_units'] ** 2)
    priors = np.array([prior_match, 1 - prior_match])

    # Train Naive Bayes classifier
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, um_param, add_one=1)

    # Calculate match probabilities
    probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, um_param, cond)
    output_prob_matrix = probability[:, 1].reshape(um_param['n_units'], um_param['n_units'])

    print(f"   ✅ Calculated probability matrix: {output_prob_matrix.shape}")
    print(f"   📈 Max probability: {np.max(output_prob_matrix):.3f}")
    print(f"   📊 Mean probability: {np.mean(output_prob_matrix):.3f}")

    ######## SIXTH CELL ###########

    # 📊 Analyze UnitMatch Results


    print("📈 Analyzing UnitMatch results...")

    # Evaluate matching performance
    match_threshold = um_param.get('match_threshold', 0.75)
    util.evaluate_output(output_prob_matrix, um_param, within_session, session_switch,
                        match_threshold=match_threshold)

    # Create binary match matrix
    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1

    # Count matches
    total_matches = np.sum(output_threshold)
    within_session_matches = np.sum(output_threshold * within_session)
    cross_session_matches = total_matches - within_session_matches

    print(f"\n🎯 Match Summary (threshold = {match_threshold}):")
    print(f"   🔗 Total matches found: {total_matches}")
    print(f"   📍 Within-session matches: {within_session_matches}")
    print(f"   🌉 Cross-session matches: {cross_session_matches}")

    # Visualize probability matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Probability matrix
    im1 = axes[0].imshow(output_prob_matrix, cmap='viridis', aspect='auto')
    axes[0].set_title('Unit Match Probability Matrix')
    axes[0].set_xlabel('Unit Index')
    axes[0].set_ylabel('Unit Index')
    plt.colorbar(im1, ax=axes[0], label='Match Probability')

    # Session boundaries
    n_units_cumsum = np.cumsum([0] + [len(units) for units in good_units])
    for boundary in n_units_cumsum[1:-1]:
        axes[0].axhline(boundary, color='red', linestyle='--', alpha=0.7)
        axes[0].axvline(boundary, color='red', linestyle='--', alpha=0.7)

    # Binary matches
    im2 = axes[1].imshow(output_threshold, cmap='Greys', aspect='auto')
    axes[1].set_title(f'Matches Above Threshold ({match_threshold})')
    axes[1].set_xlabel('Unit Index')
    axes[1].set_ylabel('Unit Index')

    # Session boundaries
    for boundary in n_units_cumsum[1:-1]:
        axes[1].axhline(boundary, color='red', linestyle='--', alpha=0.7)
        axes[1].axvline(boundary, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    print("📊 Red dashed lines show session boundaries")
    print("🎯 Diagonal represents within-session matches")
    print("🌉 Off-diagonal represents cross-session matches")

    print(um_param)

    if False: # here I have code that prints a version of the above plots with only the first four sessions, for my poster at ICAC 2025
        print("📈 Analyzing UnitMatch results...")

        # Evaluate matching performance
        match_threshold = um_param.get('match_threshold', 0.75)
        util.evaluate_output(output_prob_matrix[0:session_switch[3],0:session_switch[3]], um_param, within_session[0:session_switch[3],0:session_switch[3]], session_switch[0:3],
                            match_threshold=match_threshold)

        # Create binary match matrix
        output_threshold = np.zeros_like(output_prob_matrix[0:session_switch[3],0:session_switch[3]])
        output_threshold[output_prob_matrix[0:session_switch[3],0:session_switch[3]] > match_threshold] = 1

        # Count matches
        total_matches = np.sum(output_threshold)
        within_session_matches = np.sum(output_threshold * within_session[0:session_switch[3],0:session_switch[3]])
        cross_session_matches = total_matches - within_session_matches

        print(f"\n🎯 Match Summary (threshold = {match_threshold}):")
        print(f"   🔗 Total matches found: {total_matches}")
        print(f"   📍 Within-session matches: {within_session_matches}")
        print(f"   🌉 Cross-session matches: {cross_session_matches}")

        # Visualize probability matrix
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Probability matrix
        im1 = axes[0].imshow(output_prob_matrix[0:session_switch[4],0:session_switch[4]], cmap='viridis', aspect='auto')
        axes[0].set_title('Unit Match Probability Matrix')
        axes[0].set_xlabel('Unit Index')
        axes[0].set_ylabel('Unit Index')
        plt.colorbar(im1, ax=axes[0], label='Match Probability')

        # Session boundaries
        n_units_cumsum = np.cumsum([0] + [len(units) for units in good_units])
        for boundary in n_units_cumsum[1:4]:
            axes[0].axhline(boundary, color='red', linestyle='--', alpha=0.7)
            axes[0].axvline(boundary, color='red', linestyle='--', alpha=0.7)

        # Binary matches
        im2 = axes[1].imshow(output_threshold[0:session_switch[4],0:session_switch[4]], cmap='Greys', aspect='auto')
        axes[1].set_title(f'Matches Above Threshold ({match_threshold})')
        axes[1].set_xlabel('Unit Index')
        axes[1].set_ylabel('Unit Index')

        # Session boundaries
        for boundary in n_units_cumsum[1:4]:
            axes[1].axhline(boundary, color='red', linestyle='--', alpha=0.7)
            axes[1].axvline(boundary, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()
        from matplotlib.backends.backend_pdf import PdfPages
        pdf2Save = PdfPages(Path('F:/Jeff/Output/figures/exampleUnitMatchProbMatrix.pdf'))
        pdf2Save.savefig(fig)
        pdf2Save.close()
        # plt.show()

        print("📊 Red dashed lines show session boundaries")
        print("🎯 Diagonal represents within-session matches")
        print("🌉 Off-diagonal represents cross-session matches")

        print(um_param)


    ######### SEVENTH CELL #############


    # 🎮 Interactive Manual Curation with UnitMatch GUI


    print("🎮 Setting up interactive curation tools...")

    # Prepare data for GUI - extract all the required variables
    amplitude = extracted_wave_properties['amplitude']
    spatial_decay = extracted_wave_properties['spatial_decay']
    avg_centroid = extracted_wave_properties['avg_centroid']
    avg_waveform = extracted_wave_properties['avg_waveform']
    avg_waveform_per_tp = extracted_wave_properties['avg_waveform_per_tp']
    wave_idx = extracted_wave_properties['good_wave_idxs']
    max_site = extracted_wave_properties['max_site']
    max_site_mean = extracted_wave_properties['max_site_mean']

    # Process info for GUI
    um_gui.process_info_for_GUI(
        output_prob_matrix, match_threshold, scores_to_include, total_score,
        amplitude, spatial_decay, avg_centroid, avg_waveform, avg_waveform_per_tp,
        wave_idx, max_site, max_site_mean, waveform, within_session,
        channel_pos, clus_info, um_param
    )

    print("✅ GUI data prepared successfully")
    print("\n🎯 To launch interactive curation:")
    print("   Run: is_match, not_match, matches_GUI = um_gui.run_GUI()")
    print("\n🔍 The GUI provides:")
    print("   - Interactive visualization of unit pairs")
    print("   - Side-by-side waveform comparisons")
    print("   - Quality metric displays")
    print("   - Manual accept/reject controls")
    print("   - Real-time match probability updates")

    print("\n💡 Curation workflow:")
    print("   1. GUI shows potential matches above threshold")
    print("   2. Review waveform similarity and spatial locations")
    print("   3. Accept good matches, reject false positives")
    print("   4. Use curated results for final unit tracking")

    # Before running the GUI
    from UnitMatchPy.GUI import precalculate_all_acgs ### does not seem to actually exist?

    # Create save directory next to the first session's BombCell results
    save_dir = Path(session_results[0]['save_path']).parent / ("UnitMatch_Results_" + manualShankSplitNames[iiiii])
    save_dir.mkdir(exist_ok=True)

    # Pre-calculate and save ACGs ### acg cache seems to not be used anywhere... But I implemented a pickle loader for it.
    ACG_savepath = save_dir / Path(manualShankSplitNames[iiiii] + '_ACG.pkl')
    if ACG_savepath.exists():
       with open(ACG_savepath, 'rb') as f:
           acg_cache = pickle.load(f)
    else:
        acg_cache = precalculate_all_acgs(clus_info, um_param,save_path= ACG_savepath)



    ######### EIGHTH CELL ########
    # is_match, not_match, matches_GUI = um_gui.run_GUI() ### commented out because of a bug...
    # GUI guide: https://github.com/EnnyvanBeest/UnitMatch/blob/main/UnitMatchPy/Demo%20Notebooks/GUI_Reference_Guide.md

    ######### NINTH CELL ###########


    print("💾 Saving UnitMatch results and generating final outputs...")

    # Assign unique IDs to matched units across sessions
    UIDs = aid.assign_unique_id(output_prob_matrix, um_param, clus_info)

    if True: ### the UIDs are saved in the big old CSV, but in a way that sucks. I'll save them into a more convenient pickle here:
        good_units_listified = clus_info["good_units"]
        for (ii,unit) in enumerate(good_units_listified):
            good_units_listified[ii] = unit.tolist()

        UIDs_with_labels = {'unique_id_liberal': UIDs[0].tolist(), ### need to convert to list because of multiple python versions.
                         'unique_id': UIDs[1].tolist(),
                            'unique_id_conservative': UIDs[2].tolist(),
                            'original_unique_id': UIDs[3].tolist(),
                            'good_units_listified': good_units_listified,
                            'KS_dirs': KS_dirs
                            }
        UID_savepath_pickle = save_dir / Path('UIDs.pkl')
       #UID_savepath = save_dir / Path('UIDs.csv')
        with open(UID_savepath_pickle, 'wb') as f:
            pickle.dump(UIDs_with_labels, f)
        # import csv
        # with open(UID_savepath, "w", newline="") as f:
        #     w = csv.DictWriter(f, UIDs_with_labels.keys())
        #     w.writeheader()
        #     w.writerow(UIDs_with_labels)





    # Get final matches above threshold
    matches = np.argwhere(output_threshold == 1)



    print(f"📁 Saving results to: {save_dir}")

    # Save comprehensive UnitMatch results
    su.save_to_output(
        str(save_dir),
        scores_to_include,
        matches,
        output_prob_matrix,
        avg_centroid,
        avg_waveform,
        avg_waveform_per_tp,
        max_site,
        total_score,
        output_threshold,
        clus_info,
        um_param,
        UIDs=UIDs,
        matches_curated=None,  # Set to matches_curated if manual curation was performed
        save_match_table=True
    )

    print("✅ Results saved successfully!")

    # Generate summary statistics
    n_unique_neurons = len(np.unique(UIDs))
    n_total_units = len(np.concatenate(good_units))
    n_cross_session_matches = cross_session_matches

    print(f"\n📊 Final Results Summary:")
    print(f"{'='*60}")
    print(f"📊 Total units analyzed: {sum(len(result['unit_type']) for result in session_results)}")
    print(f"✅ Good units tracked: {n_total_units}")
    print(f"🔗 Cross-session matches found: {n_cross_session_matches}")
    print(f"🏷️  Unique neurons identified: {n_unique_neurons}")

    print(f"\n📁 Output Files Generated:")
    print(f"\n📁 Output Files Generated:")
    print(f"   📊 MatchProb.npy - Match probability matrix")
    print(f"   📋 match_table.csv - Detailed match information")

