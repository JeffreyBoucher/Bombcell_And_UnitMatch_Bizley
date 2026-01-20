############ FIRST CELL ##############


# 📦 Import Required Libraries
import numpy as np
import pandas as pd
import os as os_true
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datetime import date
from scipy import stats
from scipy.io import loadmat
import warnings
import csv
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

###### THIS IS JEFFREY AND MINE4S CODE SPECIFIC, incluse your own function that outputs a 1-dimensional array of pathlib.Path objects
# representing the directories of NP sessions, sorted in ascending order for all the sessions you want
#        of their file creation times.
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
#THIS IS JEFFREY AND MINE SPECIFIC, WILL SOON BE A FUNCTION
if True: # iterating way to get session configs for a range of sessions. Uses a copy-paste of the strategy used in spikesorting.
    ###PARAMETERS
    session_path = Path('Z:/Data/Neuropixels/F2302_Challah/')  # path to where all relevant sessions are stored
    ferret = 'F2302_Challah'
    recordingZone = 'PFC_shank0_Challah'
    behavior_path = "C:/Users/BizLab/Dropbox/Data/"+ferret

    if recordingZone == 'PFC_shank0_Challah':
        stream_id = 'imec1.ap'
        # sessionSetLabel = 'TwentiesOfMay'
        # sessionSetLabel = "AllJune"
        # sessionSetLabel = "AllMay"
        # sessionSetLabel = 'Tens_Of_June'
        sessionSetLabel = "everythingAllAtOnce"
        channel_map_to_use = 'Challah_top_PFC_shank0.imro'
        recordingZoneRepeat = 'PFC_shank0'
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
    #output_folder = Path('F:/Jeff/Output')
    output_folder = Path("E:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Output")

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
    elif sessionSetLabel == 'everythingAllAtOnce':
        sessionString = '*052024*'

    SessionsInOrder = sort_np_sessions(list(session_path.glob(sessionString)))
    sessionSetName = sessionSetLabel
    sessionNames= []
    for i, session in enumerate(SessionsInOrder):
        # if i==len(SessionsInOrder)-2:
        #     break
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
        sessionNames.append(session_name)
        session_configs[i]["name"] = session_name
        working_dir = output_folder / 'tempDir' / ferret / session_name
        dp = session_path / session_name

        probeFolder = list(dp.glob('*' + stream_id[:-3]))
        probeFolder = probeFolder[0]  # name of probe folder
        session_configs[i]["meta_file"] = probeFolder / (session_name + '_t0.' + stream_id + '.meta')

        ### hey by the way, apparently they prefer if your raw data is phase corrected and CAR. So I should maybe save some raws with SI that meet that requirement...
        session_configs[i]["raw_file"] = list(probeFolder.glob('*.bin'))[0] #obviously only works for bins, not cbin
        #session_configs[i]["ks_dir"] = output_folder / 'tempDir' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name /  Path("tempDir/kilosort4_output/sorter_output/")
        #session_configs[i]["save_path"] = output_folder / 'tempDir' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name /  Path("tempDir/kilosort4_output/sorter_output/")
        session_configs[i]["ks_dir"] = output_folder / 'tempDir' / ferret / recordingZone / recordingZoneRepeat / sessionSetName / session_name /  Path("tempDir/kilosort4_output/sorter_output/")
        session_configs[i]["save_path"] = output_folder / 'tempDir' / ferret / recordingZone / recordingZoneRepeat / sessionSetName / session_name /  Path("tempDir/kilosort4_output/sorter_output/")
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
            kilosort_version=kilosort_version,
            TestWithoutRaw=True
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

### Later used functions ###

def intersection(A,B):
    U = []
    for i in A:
        if i in B:
            U.append(i)
    return U

### GETTING ALL UNITS ###

def load_kilosort_and_bombcell_files(KS_dirs):
    spike_positions_all = []
    spike_clusters_all = []
    spike_times_all = []
    spike_amps_all = []
    nb_units_session = []
    Label_Bc_all_clus= []
    Id_Bc_all_clus = []

    time_last_spike=0
    id_last_cluster=0
    #Bombcell eliminates certain units so I'm keeping the list of KS ids of units kept by Bombcell for when I'm making the correspondance
    # with TDT corrected spike trains per unit
    id_kept_Bc= []
    session_kept_Bc= []
    print(f"📁 Kilosort directories: {len(KS_dirs)}")
    for i, ks_dir in enumerate(KS_dirs):
        print(f"   Session {i + 1}: {Path(ks_dir).name}")
        spike_positions= np.load(ks_dir / "spike_positions.npy")
        spike_clusters = np.load(ks_dir / "spike_clusters.npy")
        spike_times = np.load(ks_dir / "spike_times.npy")
        spike_amps = np.load(ks_dir / "amplitudes.npy")
        spike_temps = np.load(ks_dir / "spike_templates.npy")
        type_units = pd.read_csv(ks_dir / "cluster_bc_unitType.tsv", sep='\t')
        ids = type_units['cluster_id'].values
        labels = type_units['bc_unitType'].values
        id_kept_Bc.extend(ids)
        session_kept_Bc.extend(np.full(len(ids), i))

        #units= np.unique(spike_clusters)
        nb_units= len(ids)
        nb_units_session.append(nb_units)

        spike_times = spike_times+time_last_spike
        spike_clusters_new_ids= spike_clusters+ id_last_cluster ##indexing starts at 0
        ids = ids + id_last_cluster
        time_last_spike= max(spike_times) +30000 #maybe
        id_last_cluster= max(ids)+1

        df_spikes= pd.DataFrame(list(zip(spike_positions, spike_clusters, spike_times, spike_amps, spike_clusters_new_ids)),
                                columns=["Position", "Clus", "Time", "Amplitude", "Clus new IDs"])
        df_spikes.sort_values("Clus", ascending=True, inplace=True)

        spike_positions = list(df_spikes["Position"])
        spike_clusters = list(df_spikes["Clus"])
        spike_times = list(df_spikes["Time"])
        spike_amps = list(df_spikes["Amplitude"])
        spike_clusters_new_ids = list(df_spikes["Clus new IDs"])

        for k in range(len(spike_positions)):
            if spike_clusters[k] in type_units['cluster_id'].values:
                spike_positions_all.append(spike_positions[k])
                spike_clusters_all.append(spike_clusters_new_ids[k])
                spike_times_all.append(spike_times[k])
                spike_amps_all.append(spike_amps[k])
        for k in range(len(labels)):
            Label_Bc_all_clus.append(labels[k])
            Id_Bc_all_clus.append(ids[k])

    Df_units_kept_by_Bc= pd.DataFrame(list(zip(id_kept_Bc,session_kept_Bc)), columns=(['Id','Session']))

    spike_positions_all = np.array(spike_positions_all)
    spike_clusters_all = np.array(spike_clusters_all)
    spike_times_all = np.array(spike_times_all)
    spike_amps_all = np.array(spike_amps_all)
    Label_Bc_all_clus = np.array(Label_Bc_all_clus)
    Id_Bc_all_clus= np.array(Id_Bc_all_clus)
    GU_and_MUA= []#((Label_Bc_all_clus[i] == 'GU') or (Label_Bc_all_clus[i] == 'MUA')) for i in range(len(Id_Bc_all_clus))]

    count=0
    for i,type_clus in enumerate(Label_Bc_all_clus):
        if (type_clus == 'GOOD') or (type_clus == 'MUA'):
            GU_and_MUA.append(count)
            count+=1
        else:
            GU_and_MUA.append(float('nan'))
    return spike_positions_all, spike_clusters_all, spike_times_all, spike_amps_all, nb_units_session, id_kept_Bc, GU_and_MUA, Id_Bc_all_clus

def create_list_all_units(spike_positions_all, spike_clusters_all, spike_times_all, spike_amps_all):
    """
    Separates the info (spike times and positions) of each cluster
    :param spike_positions_all: all spike posititions ORDDERED BY CLUSTER ID
    :param spike_times_all: all spike times ORDDERED BY CLUSTER ID
    :param spike_amps_all: all spike amplitudes ORDDERED BY CLUSTER ID
    :param spike_clusters_al: all spike cluster
    :return:
    spike_pos_by_clus, spike_time_by_clus, spike_time_by_clus: list of list of spike times/posititions/amplitudes by clus, size= (nb_clus, for each list the nb of spike by cluster)
    WARNING: for spike pos, there is x and y, not just one number
    original_clus_id: list of the origals cluster ids of size n_clusters (usefull later to keep correspondance between positions of cluster in list and cluster original id
    """
    spike_pos_by_clus= []
    spike_time_by_clus= []
    spike_amps_by_clus= []
    original_clus_id= []
    clus=spike_clusters_all[0]
    spike_for_this_clus = []
    times_for_this_clus = []
    amps_for_this_clus = []
    count=0
    for i, pos in enumerate(spike_positions_all):
        if spike_clusters_all[i] == clus:
            spike_for_this_clus.append(pos)
            times_for_this_clus.append(spike_times_all[i])
            amps_for_this_clus.append(spike_amps_all[i])

        else:
            spike_pos_by_clus.append(spike_for_this_clus.copy())
            spike_time_by_clus.append(times_for_this_clus.copy())
            spike_amps_by_clus.append(amps_for_this_clus.copy())
            original_clus_id.append(clus)
            spike_for_this_clus=[]
            times_for_this_clus=[]
            amps_for_this_clus=[]
            spike_for_this_clus.append(pos)
            times_for_this_clus.append(spike_times_all[i])
            amps_for_this_clus.append(spike_amps_all[i])
            clus= spike_clusters_all[i]
    spike_pos_by_clus.append(spike_for_this_clus.copy())
    spike_time_by_clus.append(times_for_this_clus.copy())
    spike_amps_by_clus.append(amps_for_this_clus.copy())
    original_clus_id.append(clus)
    return spike_pos_by_clus, spike_time_by_clus, spike_amps_by_clus, original_clus_id


def correct_motion_on_spike_pos(position, motion_sessions, session, half_bin_size=50):
    if session== 0:
        return position
    else:
        new_position = position
        for i in range(session):
            motion_df = motion_sessions[motion_sessions['session'] ==i]
            bins_center= np.array(motion_df['center_space_bin'])
            bin_low= list(np.where(new_position <= bins_center+ half_bin_size)) #we look for the indice of the
            bin_high = list(np.where(new_position >= bins_center- half_bin_size))
            if len(bin_low[0])!=0 and len(bin_high[0])!=0:
                if len(intersection(bin_low[0],bin_high[0])) == 1: #when computing new bins centers for weeks, distance between bins changes
                    bin_loc= intersection(bin_low[0],bin_high[0])[0]
                else:
                    dist_low= abs(new_position-bins_center[bin_low[0][0]])
                    dist_high= abs(bins_center[bin_high[0][-1]]-new_position)
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
            new_position+= local_motion
        return new_position


def create_dataframe_of_cluster(spike_pos_by_clus, spike_time_by_clus, spike_amps_by_clus, original_clus_id,
                                nb_units_session, motion_sessions, id_kept_Bc):
    """Creates data frame of clusters with columns:
    "Original kilosort id", "Positions", "Times", "Amplitudes", "Session",
    "Mean amplitude", "Mean y position", "Variance y position", "Mean corrected y position"
    """
    dict_of_clus= {"Original kilosort id": original_clus_id,
        "Positions": spike_pos_by_clus,
        "Times": spike_time_by_clus,
        "Amplitudes": spike_amps_by_clus}
    Clusters= pd.DataFrame(dict_of_clus)
    session_clus= []
    session_here= 0
    for i in nb_units_session: #um_param['n_units_per_session']: #Look at number of ALL units
        for j in range(i):
            session_clus.append(session_here)
        session_here+=1
    Clusters["Session"]= session_clus
    mean_amps=[]
    for i in Clusters["Amplitudes"].to_list():
        mean_amps.append(np.mean(i))
    Clusters["Mean amplitude"]= mean_amps
    y_pos=[]
    x_pos=[]
    for i in Clusters["Positions"].to_list():
        x_pos.append(np.array(i)[:,0])
        y_pos.append(np.array(i)[:,1])
    Clusters["X Positions"]= x_pos
    Clusters["Y Positions"]= y_pos
    mean_pos=[]
    for i in Clusters["Y Positions"].to_list():
        mean_pos.append(np.mean(i))
    Clusters["Mean y position"]= mean_pos
    var_pos=[]
    for i in Clusters["Y Positions"].to_list():
        var_pos.append(np.std(i))
    Clusters["Variance y position"]= var_pos
    motion_corrected_pos=[]
    for i,pos in enumerate(Clusters["Mean y position"].to_list()):
        motion_corrected_pos.append(
            correct_motion_on_spike_pos(pos,motion_sessions,
                                        Clusters["Session"].iloc[i], half_bin_size=50)  )
    Clusters["Mean corrected y position"]=  motion_corrected_pos
    corr_pos_array=[]
    for i,pos in enumerate(Clusters["Y Positions"].to_list()):
        u=len(pos)
        corr_pos_array.append(np.full(u, Clusters['Mean corrected y position'].iloc[i]))
    Clusters["Array new pos"]= corr_pos_array
    corr_pos_again=[]
    for i,pos in enumerate(Clusters["Y Positions"].to_list()):
        u=len(pos)
        corr_pos_again.append(np.full(u, Clusters['Mean y position'].iloc[i]))
    Clusters["Array old pos"]= corr_pos_again
    Clusters["Ori ID Bc per session"]= id_kept_Bc
    return Clusters

### FILTERING FOR GOOD AND MUA + ADDING UNIT MATCH INFO ###

def filter_GU_MUA(Clusters, GU_and_MUA, Id_Bc_all_clus):
    ori_ids=[]
    for i,id in enumerate(Id_Bc_all_clus):
        if type(GU_and_MUA[i]) == int:
            ori_ids.append(id)
    Clusters_GU_and_MUA= Clusters[Clusters["Original kilosort id"].isin(ori_ids)]
    return Clusters_GU_and_MUA

def get_UM_id(Clusters, UID):
    if Clusters["Original kilosort id"].shape[0] != len(UID[0]):
        print("Unit Match IDs do not match Good and Multi Units")
    else:
        New_df_Clusters= Clusters
        New_df_Clusters["Original Unit Match ID"]= UID[3]
        New_df_Clusters["Conservative Unit Match ID"]= UID[2]
        New_df_Clusters["Intermediary Unit Match ID"]= UID[1]
        New_df_Clusters["Liberal Unit Match ID"]= UID[0]
        return New_df_Clusters

def check_matching_status(Clusters):
    non_matched_units=[]
    matched_within=[]
    matched_accross=[]
    matched_both_ways=[]
    units= np.unique(np.array(Clusters["Conservative Unit Match ID"]))
    print(f"There are {len(units)} units overall")
    for i in units:
        Cluster= Clusters[Clusters["Conservative Unit Match ID"] ==i]
        if Cluster.shape[0]==1:
            non_matched_units.append(i)
        else:
            sessions_unit=Cluster["Session"].value_counts()
            print("Break")
            if sessions_unit[sessions_unit >1].shape[0] ==0:
                matched_accross.append(i)
            else:
                if sessions_unit.shape[0] ==1:
                    matched_within.append(i)
                else:
                    matched_both_ways.append(i)
    return non_matched_units, matched_within, matched_accross, matched_both_ways

def add_matching_status_to_df(Clusters, non_matched_units, matched_within, matched_accross, matched_both_ways):
    matching_status=[]
    for i in Clusters["Conservative Unit Match ID"]:
        if i in non_matched_units:
            matching_status.append('Not matched')
        elif i in matched_within:
            matching_status.append('Within')
        elif i in matched_accross:
            matching_status.append('Accross')
        elif i in matched_both_ways:
            matching_status.append('Both ways')
    Clusters["Matching status"]= matching_status

### ADDING INFO ABOUT CLUSTER TO DATA FRAME  ###
def add_var_pos(Clusters):
    position= Clusters['Y Positions'].to_list()
    var_pos= []
    for i in position:
        var_pos.append(np.std(i))
    Clusters['Variance Y Positions'] = var_pos

def add_isi(Clusters):
    times = Clusters['Times'].to_list()
    isi= []
    for i,spike_times in enumerate(times):
        isi_this_clus= []
        for j in range(len(spike_times)-1):
            isi_this_clus.append(spike_times[j+1]-spike_times[j])
        isi.append(isi_this_clus)
    Clusters['ISI']= isi

def add_average_firing_rate(Clusters):
    times = Clusters['Times'].to_list()
    firing_rates= []
    for i,spike_times in enumerate(times):
        nb_spikes= len(spike_times)
        session_length= spike_times[-1]- spike_times[0]
        firing_rates.append(nb_spikes/session_length)
    Clusters['Average firing rate']= firing_rates

def get_motion_correction_from_all_positions(positions, motion_sessions, session, new_position):
    positions_this_clus= np.array(positions)
    positions_this_clus_new = positions_this_clus#[:,1]#.copy()
    #positions_this_clus = np.vstack(positions_this_clus_new)
    nb_spike = np.shape(positions_this_clus)[0]
    motion_corrected_clus= new_position
    motion_corrected_clus = np.full(nb_spike, motion_corrected_clus)
    return motion_corrected_clus

def get_corrected_average_positions(Clusters, motion_sessions):
    Clusters["Corrected average position"]= get_motion_correction_from_all_positions(Clusters["Y Positions"], motion_sessions, Clusters["Session"], Clusters['Mean corrected y position'])

def get_average_position_list(Cluster):
    positions_this_clus= np.array(positions)
    positions_this_clus_new = positions_this_clus[:,1]#.copy()
    positions_this_clus = np.vstack(positions_this_clus_new)
    nb_spike = np.shape(positions_this_clus)[0]
    motion_corrected_clus= new_position
    motion_corrected_clus = np.full(nb_spike, motion_corrected_clus)

def get_quantile_amplitude_per_session(Clusters):
    sessions= np.unique(np.array(Clusters['Session']))
    first_quantiles= []
    third_quantiles= []
    for i in sessions:
        Session= Clusters[Clusters['Session'] == i]
        Q1= Session['Mean amplitude'].quantile(0.25)
        first_quantiles.append(Q1)
        Q3= Session['Mean amplitude'].quantile(0.75)
        third_quantiles.append(Q3)
    df_quantiles= pd.DataFrame({'Session': sessions, '1rst Q': first_quantiles, '3rd Q': third_quantiles})
    all_Q1= []
    all_Q3= []
    for i in range(Clusters.shape[0]):
        sess= Clusters['Session'].iloc[i]
        all_Q1.append(df_quantiles[df_quantiles['Session'] ==sess]['1rst Q'].to_list()[0])
        all_Q3.append(df_quantiles[df_quantiles['Session'] ==sess]['3rd Q'].to_list()[0])
    Clusters['1rst Q amplitude'] = all_Q1
    Clusters['3rd Q amplitude'] = all_Q3

### LOADING BEHAVIORAL DATA FRAME ###


def get_data_from_mat_file(mdata, variables, file):
    Sorted_data = pd.DataFrame(columns=variables)
    nbTrials = len(mdata[0])
    testvariable_presence = True
    for i in variables: #checking whether all the variables exist in the data list
            if not(i in mdata.dtype.names):
                testvariable_presence = False
                var = i
    if not(testvariable_presence): #if there's at least one missing variable
        print('Missing variable'+var+' :'+file)
    else:
        for i in variables:
            testvariable_empty = False
            testonlyoneelement = False
            if len(mdata[i][0][0][0]) == 0:
                testvariable_empty = True
            if len(mdata[i][0][0][0]) == 1:
                testonlyoneelement = True
            if not(testvariable_empty) and testonlyoneelement:
                Sorted_data[i]= [mdata[i][0][n][0][0] for n in range(nbTrials)]
            if not(testvariable_empty) and not(testonlyoneelement):
                Sorted_data[i]= [mdata[i][0][n][0] for n in range(nbTrials)]
    return Sorted_data

def sort_date(directory, date_min=None, date_max= None):
    new_list_dir = []
    dates_kept_files = []
    for i,file in enumerate(directory):
        if str(file)[-4:] != ".mat":
            continue
        my_date= str(file).split(' ')[0]
        year_file= int(my_date[-4:])
        if len(my_date) == 8:
            day_file= int(my_date[0])
            month_file= int(my_date[2])
        elif len(my_date) == 10:
            day_file= int(my_date[:2])
            month_file= int(my_date[3:5])
        else:
            if my_date[1]=='_':
                day_file= int(my_date[0])
                month_file= int(my_date[2:4])
            else:
                day_file= int(my_date[:2])
                month_file= int(my_date[3])
        print('break')
        creation_date= date(year_file, month_file,day_file)
        if date_min==None and date_max==None:
            new_list_dir.append(file)
            dates_kept_files.append(creation_date)
        elif date_min==None:
            if creation_date < date_max:
                new_list_dir.append(file)
                dates_kept_files.append(creation_date)
        elif date_max==None:
            if creation_date > date_min:
                new_list_dir.append(file)
                dates_kept_files.append(creation_date)
        else:
            if creation_date > date_min and creation_date < date_max:
                new_list_dir.append(file)
                dates_kept_files.append(creation_date)
    df_file_and_dates= pd.DataFrame(list(zip(new_list_dir, dates_kept_files)), columns= ['file', 'creation date'])
    df_file_and_dates.sort_values('creation date')
    return df_file_and_dates['file'].to_list()

def check_dates(file, date_min=None, date_max= None):
    my_date= str(file).split(' ')[0]
    year_file= int(my_date[-4:])
    if len(my_date) == 8:
        day_file= int(my_date[0])
        month_file= int(my_date[2])
    elif len(my_date) == 10:
        day_file= int(my_date[:2])
        month_file= int(my_date[3:5])
    else:
        if my_date[1]=='_':
            day_file= int(my_date[0])
            month_file= int(my_date[2:4])
        else:
            day_file= int(my_date[:2])
            month_file= int(my_date[3])
    print('break')
    creation_date= date(year_file, month_file,day_file)
    if date_min==None and date_max==None:
        return True
    elif date_min==None:
        if creation_date < date_max:
            return True
        else:
            return False
    elif date_max==None:
        if creation_date > date_min:
            return True
        else:
            return False
    else:
        if creation_date > date_min and creation_date < date_max:
            return True
        else:
            return False

def load_behavioral_data(path, date_min= None, date_max= None):

    interest_variables= ['response','correctionTrial','catchTrial',
                         'talker', 'distractors','dDurs',
                         'startTrialLick', 'lickRelease','timeToTarget',
                         'absentTime',
                         'distractorTimes', 'lastDistractor', 'reactionTime','presentedDistractors']#,'fs']
    interest_variables_already_there= ['response','correctionTrial','catchTrial',
                         'talker', 'distractors','dDurs',
                         'startTrialLick', 'lickRelease','timeToTarget',
                         'absentTime']#,'fs']

    data_frame_behavior= pd.DataFrame(columns= interest_variables)
    directory = os_true.listdir(path)
    directory= sort_date(directory, date_min=date_min, date_max= date_max)
    session= []
    for i,file in enumerate(directory):
        if file[-4:] == '.mat' and check_dates(file, date_min): #verifying if it is really a mat file and a training file =date(2025,11,11)
            data = loadmat(path+"/"+file)
            if sorted(data.keys())[3] == 'data' :
                mdata =data['data']
                df = get_data_from_mat_file(mdata, interest_variables_already_there,file)
                len_trial= []
                for j,start_trial in enumerate(df['startTrialLick'].to_list()):
                    if j == len(df['startTrialLick'].to_list())-1:
                        len_trial.append(df['lickRelease'].to_list()[j] - start_trial) ### TO CHANGE TO A BETTER END OF TRIAL VARIABLE
                        break
                    len_trial.append(df['startTrialLick'].to_list()[j+1] - start_trial)
                df['TrialLength'] = len_trial
                data_frame_behavior= pd.concat([data_frame_behavior, df])
                session.extend(np.full(df.shape[0], i))
    data_frame_behavior['Session']= session

    return data_frame_behavior

def get_min_and_max_dates(list_of_bhv_file_names):
    "This function takes as argument a list of behavioral file names and returns the date of the oldest and most recent of those files"
    list_dates=[]
    for i,file in enumerate(list_of_bhv_file_names):
        my_date= str(file).split('_')[0]
        year_file= int(my_date[-4:])
        day_file= int(my_date[0:2])
        month_file= int(my_date[2:4])
        creation_date= date(year_file, month_file,day_file)
        list_dates.append(creation_date)
    date_min= min(list_dates)
    date_max= max(list_dates)
    return date_min, date_max

def get_df_behavior(path, neuropixel_rec_sessions):
    date_min, date_max= get_min_and_max_dates(neuropixel_rec_sessions)
    return load_behavioral_data(path, date_min, date_max)


# CONNECTING NEURAL AND BEHAVIORAL DATA FUNCTIONS

def align_times_stim(times, events, window=[-1, 1]):
    """
    Aligns times to events.
    FROM JULES/JEFF CODE MODIFIED BY ME
    Parameters
    ----------
    times : np.array
        Spike times (in seconds).
    events : np.array
        Event times (in seconds).
    window : list
        Window around event (in seconds).

    Returns
    -------
    aligned_times : np.array
        Aligned spike times.
    """

    t = np.array(np.sort(times)[0])
    aligned_times = []
    for i, e in enumerate(events):
        ts = t - e  # ts: t shifted (times relative to event
        tsc = ts[(ts >= window[0]) & (
                    ts <= window[1])]  # tsc: ts clipped (times of spikes in window around event relative to event time)
        # al_t = np.full((tsc.size, 2), i, dtype='float')  # creates 2D array where first line is i
        # #al_t[:,1] = stim[i] remain from Jules code --> id type of stim for euclidian decoding
        # al_t[:, 1] = tsc  # second line is tsc
        aligned_times.append(list(tsc))
    return aligned_times
#def filter_behavior_function(Df_Behavior, filter):


### PLOTTING CODES ###

def plot_spikes_matchingunits(spike_pos_all_units, spike_times_all_units, Clusters):
    import matplotlib
    import matplotlib.colors as mcolors
    matplotlib.use('TkAgg')
    colors= {'Not matched':'red', 'Within':'green', 'Accross':'blue', 'Both ways':'orange'}
    #colortable= list(mcolors.TABLEAU_COLORS)
    spike_pos_all_units= np.array(spike_pos_all_units)
    spike_times_all_units= np.array(spike_times_all_units)
    typeunits= np.array(Clusters["Matching status"])
    fig, ax = plt.subplots()
    for i in range(len(spike_pos_all_units)):
        color= colors[typeunits[i]]
        ax.scatter( spike_times_all_units[i], spike_pos_all_units[i], s=5, color = color)#marker="o", color="blue", markersize= 5)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Position")
    plt.show()

def plot_spikes_categorical(spike_pos_all_units, spike_times_all_units, type_units):
    import matplotlib
    import matplotlib.colors as mcolors
    matplotlib.use('TkAgg')
    all_matched_units= np.unique(type_units)
    colors=dict()
    load_colors= list(mcolors.XKCD_COLORS.values())
    for i in all_matched_units:
        colors[i] = load_colors[i]
    #colortable= list(mcolors.TABLEAU_COLORS)
    fig, ax = plt.subplots()
    for i, id in enumerate(type_units):
        color= colors[id]
        ax.scatter( spike_times_all_units[i], spike_pos_all_units[i], s=5, color = color)#marker="o", color="blue", markersize= 5)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Position")
    plt.show()

def plot_spikes_gradient_accross_all_sessions(spike_pos_all_units, spike_times_all_units, typeunits):
    import matplotlib
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    matplotlib.use('TkAgg')
    spike_pos_all_units= np.array(spike_pos_all_units)
    spike_times_all_units= np.array(spike_times_all_units)
    typeunits= np.array(typeunits)
    typeunits= (typeunits-min(typeunits))/(max(typeunits)-min(typeunits))
    viridis = matplotlib.colormaps['viridis'].resampled(8)
    #cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)))
    #colortable= list(mcolors.TABLEAU_COLORS)
    fig, ax = plt.subplots()
    for i in range(len(spike_pos_all_units)):
        ax.scatter( spike_times_all_units[i], spike_pos_all_units[i], s=5, color = viridis(typeunits[i]))#, cmap= cmap)#marker="o", color="blue", markersize= 5)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Position")
    plt.show()

def plot_spikes_gradient_per_session(filtered_df, gradient_parameters, pos_to_plot):
    #per example 'Mean amplitude'
    import matplotlib
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    matplotlib.use('TkAgg')
    spike_pos_all_units= np.array(filtered_df[pos_to_plot])
    spike_times_all_units= np.array(filtered_df['Times'])
    map_colors_per_session=[]
    for i in np.unique(sessions):
        Df_this_session= filtered_df[filtered_df['Session'] == i]
        this_session_gradient= (Df_this_session[gradient_parameters]-min(Df_this_session[gradient_parameters]))/(max(Df_this_session[gradient_parameters])-min(Df_this_session[gradient_parameters]))
        map_colors_per_session.append(this_session_gradient)
    typeunits= np.concatenate(map_colors_per_session)
    viridis = matplotlib.colormaps['viridis'].resampled(8)
    #cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)))
    #colortable= list(mcolors.TABLEAU_COLORS)
    fig, ax = plt.subplots()
    for i in range(len(spike_pos_all_units)):
        ax.scatter( spike_times_all_units[i], spike_pos_all_units[i], s=5, color = viridis(typeunits[i]))#, cmap= cmap)#marker="o", color="blue", markersize= 5)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Position")
    plt.show()

def plot_hist(X, ax, variable ='ISI', bins=20):
    import matplotlib
    import matplotlib.colors as mcolors
    matplotlib.use('TkAgg')
    colortable= list(mcolors.TABLEAU_COLORS)
    #fig, ax = plt.subplots()
    ax.hist(X, bins= bins)
    ax.set_xlabel(variable)
    ax.set_ylabel(f"Count")
    #plt.show()

def plot_hist_matching_units(Clusters, param='ISI'):
    units= np.unique(np.array(Clusters['Conservative Unit Match ID']))
    for i in units:
        df_this_unit= Clusters[Clusters['Conservative Unit Match ID'] == i]
        nb_units= df_this_unit.shape[0]
        fig, axs = plt.subplots(2, nb_units, figsize=(14, 16))
        for j in range(nb_units):
            plot_hist(df_this_unit[param].iloc[j], axs[0,j])
        ISI_whole_unit= np.concatenate(np.array(df_this_unit[param]))
        plot_hist(ISI_whole_unit, axs[1, 0])
        plt.show()

def plot_raster_minimal(times_aligned_to_event, Trials_alignement_variable, session, unit_UM, unit_KS, ax):
    for j in range(len(times_aligned_to_event)):
        ax.scatter( times_aligned_to_event[j], np.full(len(times_aligned_to_event[j]),j), s=5, color="blue")#marker="o", color="blue", markersize= 5)
    ax.axvline(x=0, color='r', label=f'Time of {Trials_alignement_variable}')
    ax.set_xlabel(f"Time relative to {Trials_alignement_variable}")
    ax.set_ylabel(f"Trial number")
    ax.set_title(f"Raster of unit {unit_UM}, present in session {session},\nof original Kilosort id {unit_KS}")


def plot_rasters_units(Clusters, Behavior, Trials_alignement_variable= 'startTrialLick', Trial_sorting_filter= None,
                       figsize=(16, 8)):
    all_units_here= np.unique(Clusters['Conservative Unit Match ID'])
    #import matplotlib
    #matplotlib.use('TkAgg')
    for i,unit in enumerate(all_units_here):
        ThisUnit= Clusters[Clusters['Conservative Unit Match ID'] == unit]
        eliminate_annoying_units= np.concatenate(np.array(ThisUnit['Time corrected to TDT']))
        if '...' in eliminate_annoying_units:
            continue
        if ThisUnit['Session'].iloc[0]>0:
            print('BReak')
        sessions_this_unit= np.unique(ThisUnit['Session'])
        list_around_event_spiketrains= []
        if len(sessions_this_unit) ==0:
            continue
        elif len(sessions_this_unit) == 1:
            session= np.unique(ThisUnit['Session'])[0]
            BehaviorThisSession= Behavior[Behavior['Session']== session]
            ThisUnitThisSession= ThisUnit[ThisUnit['Session']== session]
            if Trial_sorting_filter is not None:
                BehaviorThisSession = filter_behavior_function(BehaviorThisSession, Trial_sorting_filter)
            nb_ori_units= len(np.unique(ThisUnitThisSession['Original kilosort id']))
            if nb_ori_units ==0:
                continue
            if nb_ori_units==1:
                event_times_this_session= np.array(BehaviorThisSession[Trials_alignement_variable])
                spike_times_this_unit= np.array(ThisUnitThisSession['Time corrected to TDT'])
                times_aligned_to_event= align_times_stim(spike_times_this_unit, event_times_this_session, window=[-1, 2])
                unit_KS= ThisUnitThisSession['Original kilosort id']
                fig, ax = plt.subplots()
                plot_raster_minimal(times_aligned_to_event, Trials_alignement_variable, session, unit, unit_KS, ax)
                fig.show()
                continue
            if nb_ori_units>1:
                spiketrains_all_subunits= []
                fig, axs = plt.subplots(2, nb_ori_units, figsize=figsize)
                for j, unit_KS in enumerate(ThisUnitThisSession['Original kilosort id']):
                    ThisUnitThisSessionKS= ThisUnitThisSession[ThisUnitThisSession['Original kilosort id'] == unit_KS]
                    event_times_this_session= np.array(BehaviorThisSession[Trials_alignement_variable])
                    spike_times_this_unit= np.array(ThisUnitThisSessionKS['Time corrected to TDT'])
                    times_aligned_to_event= align_times_stim(spike_times_this_unit, event_times_this_session, window=[-1, 2])
                    plot_raster_minimal(times_aligned_to_event, Trials_alignement_variable, session, unit, unit_KS, axs[0,j])
                    if not j:
                        spiketrains_all_subunits= times_aligned_to_event.copy()
                    else:
                        spiketrains_all_subunits.extend(times_aligned_to_event)
                plot_raster_minimal(spiketrains_all_subunits, Trials_alignement_variable,
                                    session, unit, ThisUnitThisSession['Original kilosort id'], axs[1,0])
                fig.show()
                continue
        elif ThisUnit['Matching status'].iloc[0] == "Accross":
            ori_units= np.unique(ThisUnit['Original kilosort id'])
            spiketrains_all_subunits= []
            fig, axs = plt.subplots(2, len(ori_units), figsize=figsize)
            for j, unit_KS in enumerate(ori_units):
                ThisUnitKS= ThisUnit[ThisUnit['Original kilosort id'] == unit_KS]
                session_this_unit= ThisUnitKS['Session'].iloc[0]
                BehaviorThisSession= Behavior[Behavior['Session']== session_this_unit]
                if Trial_sorting_filter is not None:
                    BehaviorThisSession = filter_behavior_function(BehaviorThisSession, Trial_sorting_filter)
                event_times_this_session= np.array(BehaviorThisSession[Trials_alignement_variable])
                spike_times_this_unit= np.array(ThisUnitKS['Time corrected to TDT'])
                times_aligned_to_event= align_times_stim(spike_times_this_unit, event_times_this_session, window=[-1, 2])
                plot_raster_minimal(times_aligned_to_event, Trials_alignement_variable,
                                    session_this_unit, unit, unit_KS, axs[0,j])
                if not j:
                    spiketrains_all_subunits.extend(times_aligned_to_event.copy())
                else:
                    spiketrains_all_subunits.extend(times_aligned_to_event.copy())
            plot_raster_minimal(spiketrains_all_subunits, Trials_alignement_variable,
                                sessions_this_unit, unit, ori_units, axs[1,0])
            fig.show()
            continue
        elif ThisUnit['Matching status'].iloc[0] == "Both ways":
            spiketrains_all_subunits= []
            ori_units= np.unique(ThisUnit['Original kilosort id'])
            fig, axs = plt.subplots(3, len(ori_units), figsize=figsize)
            count_units=0
            for j,session in enumerate(sessions_this_unit):
                BehaviorThisSession= Behavior[Behavior['Session']== session]
                ThisUnitThisSession= ThisUnit[ThisUnit['Session']== session]
                if Trial_sorting_filter is not None:
                    BehaviorThisSession = filter_behavior_function(BehaviorThisSession, Trial_sorting_filter)
                nb_ori_units= len(np.unique(ThisUnitThisSession['Original kilosort id']))
                if nb_ori_units ==0:
                    continue
                if nb_ori_units==1:
                    event_times_this_session= np.array(BehaviorThisSession[Trials_alignement_variable])
                    spike_times_this_unit= np.array(ThisUnitThisSession['Time corrected to TDT'])
                    times_aligned_to_event= align_times_stim(spike_times_this_unit, event_times_this_session, window=[-1, 2])
                    unit_KS= ThisUnitThisSession['Original kilosort id']
                    plot_raster_minimal(times_aligned_to_event, Trials_alignement_variable, session, unit, unit_KS, axs[0,count_units])
                    if not j and not count_units:
                        spiketrains_all_subunits= times_aligned_to_event
                    else:
                        spiketrains_all_subunits.extend(times_aligned_to_event.copy())
                    count_units+=1
                if nb_ori_units>1:
                    spiketrains_all_subunits_this_session= []
                    for k, unit_KS in enumerate(ThisUnitThisSession['Original kilosort id']):
                        ThisUnitThisSessionKS= ThisUnitThisSession[ThisUnitThisSession['Original kilosort id'] == unit_KS]
                        event_times_this_session= np.array(BehaviorThisSession[Trials_alignement_variable])
                        spike_times_this_unit= np.array(ThisUnitThisSessionKS['Time corrected to TDT'])
                        times_aligned_to_event= align_times_stim(spike_times_this_unit, event_times_this_session, window=[-1, 2])
                        plot_raster_minimal(times_aligned_to_event, Trials_alignement_variable, session, unit, unit_KS, axs[0,k])
                        if not j and not count_units:
                            spiketrains_all_subunits= times_aligned_to_event.copy()
                            spiketrains_all_subunits_this_session= times_aligned_to_event.copy()
                        else:
                            spiketrains_all_subunits.extend(times_aligned_to_event)
                            spiketrains_all_subunits_this_session.extend(times_aligned_to_event.copy())
                    plot_raster_minimal(spiketrains_all_subunits_this_session, Trials_alignement_variable,
                                        session, unit, ThisUnitThisSession['Original kilosort id'], axs[1,j])
            plot_raster_minimal(spiketrains_all_subunits, Trials_alignement_variable,
                                        sessions_this_unit, unit, ori_units, axs[2,0])
            fig.show()
            continue


def correct_motion_on_channel_pos(positions, motion_week, motion_sessions, session, week_session_correspondance, half_bin_size=50):
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
                bin_low= list(np.where(pos <= bins_center+ half_bin_size)) #we look for the indice of the
                bin_high = list(np.where(pos >= bins_center- half_bin_size))
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
                new_positions.append(pos + local_motion)
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
                bin_low= list(np.where(pos <= bins_center+ half_bin_size)) #we look for the indice of the
                bin_high = list(np.where(pos >= bins_center- half_bin_size))
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
                new_positions.append(pos + local_motion)
            y_positions_init= new_positions.copy()
        for i in range(week_sessions,session):
            new_positions = []
            motion_df = motion_sessions[motion_sessions['session'] ==i]
            for j,pos in enumerate(y_positions_init):
                bins_center= np.array(motion_df['center_space_bin'])
                bin_low= list(np.where(pos <= bins_center+ half_bin_size)) #we look for the indice of the
                bin_high = list(np.where(pos >= bins_center- half_bin_size))
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
                new_positions.append(pos + local_motion)
            y_positions_init= new_positions.copy()
        print(f'{j}')
        all_positions = np.array([list(x_positions), list(new_positions)])
        all_positions = all_positions.T
        return all_positions

def get_df_and_plots(Df_clusters, GU_and_MUA, Id_Bc_all_clus, UIDs):

    #plot_spikes_matchingunits(Df_Matched_GU_MUA['Y Positions'], Df_Matched_GU_MUA['Times'])
    plot_spikes_matchingunits(Df_Matched_GU_MUA['Array old pos'], Df_Matched_GU_MUA['Times'], Df_Matched_GU_MUA)
    #plot_spikes_matchingunits(Df_Matched_GU_MUA['Array new pos'], Df_Matched_GU_MUA['Times'])


    plot_spikes_gradient_accross_all_sessions(Df_Matched_GU_MUA['Y Positions'], Df_Matched_GU_MUA['Times'], Df_Matched_GU_MUA['Mean amplitude'])
    plot_spikes_gradient_accross_all_sessions(Df_Matched_GU_MUA['Y Positions'], Df_Matched_GU_MUA['Times'], Df_Matched_GU_MUA['Average firing rate'])


    plot_spikes_gradient_per_session(Df_Matched_GU_MUA, 'Mean amplitude', 'Array old pos')
    plot_spikes_gradient_per_session(Df_Matched_GU_MUA, 'Mean amplitude', 'Array new pos')
    plot_spikes_gradient_per_session(Df_Matched_GU_MUA, 'Average firing rate', 'Array old pos')
    plot_spikes_gradient_per_session(Df_Matched_GU_MUA, 'Average firing rate', 'Array new pos')

    first_quantile_amp= Df_Matched_GU_MUA[Df_Matched_GU_MUA['Mean amplitude'] - Df_Matched_GU_MUA['1rst Q amplitude']]
    third_quantile_amp= Df_Matched_GU_MUA[Df_Matched_GU_MUA['Mean amplitude'] > Df_Matched_GU_MUA['3rd Q amplitude']]

    plot_spikes_gradient_per_session(first_quantile_amp, 'Mean amplitude', 'Array old pos')
    plot_spikes_gradient_per_session(first_quantile_amp, 'Mean amplitude', 'Array new pos')

    plot_spikes_gradient_per_session(third_quantile_amp, 'Mean amplitude', 'Array old pos')
    plot_spikes_gradient_per_session(third_quantile_amp, 'Mean amplitude', 'Array new pos')

    matched_accross= Df_Matched_GU_MUA[Df_Matched_GU_MUA['Matching status'] == 'Accross']
    plot_hist_matching_units(matched_accross, param='ISI')


print("🎯 Setting up UnitMatch for cross-session tracking...")

# Get default UnitMatch parameters
um_param = default_params.get_default_param()

#Set up parameters that define whether the whole code is ran or variables are just loaded from pickles
RunUnitMatch= False
SaveUIDToCSV= False
LoadPickleBombcellAndKilosort= True
LoadPickleDataFrame= False
LoadTDTCorrectedTimes= True
ApplyMotionCorrectionInUnitMatch= True


# Set up paths from our BombCell results
KS_dirs = [result['ks_dir'] for result in session_results]
um_param['KS_dirs'] = KS_dirs

#### I set up the "custom_bombcell_paths" in cell 2 now.

print(f"📁 Kilosort directories: {len(KS_dirs)}")

print(f"📊 BombCell unit classifications:")
for i, bc_path in enumerate(custom_bombcell_paths):
    exists = "✅" if Path(bc_path).exists() else "❌"
    print(f"   Session {i + 1}: {exists} {Path(bc_path).name}")

print(f"🎯 Raw waveforms for UnitMatch:")
for i, wv_path in enumerate(custom_raw_waveform_paths):
    exists = "✅" if Path(wv_path).exists() else "❌"
    n_files = len(list(Path(wv_path).glob("Unit*_RawSpikes.npy"))) if Path(wv_path).exists() else 0
    print(f"   Session {i + 1}: {exists} {n_files} waveform files")

motion_sessions= pd.read_csv(KS_dirs[-1]/ 'motion_sessions.csv')
motion_sessions_old= pd.read_csv(KS_dirs[-1]/ 'motion_sessions_old.csv')
motion_week= pd.read_csv(KS_dirs[-1]/ 'motion_weeks.csv')
week_session_correspondance= np.load(KS_dirs[-1]/ 'session_to_week_id.npy')



if RunUnitMatch:
# Setup UnitMatch paths - this matches exactly what processing_playground does
    try:
        wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(  #
            KS_dirs,
            custom_raw_waveform_paths=custom_raw_waveform_paths,
            custom_bombcell_paths=custom_bombcell_paths,
            motion_week=motion_week,
            motion_sessions=motion_sessions,
            week_session_correspondance=week_session_correspondance,
            do_channel_correction=ApplyMotionCorrectionInUnitMatch
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

        #CREATING A LIST OF MATCHED UNITS NEW ID, ORIGINAL ID AND ORIGINAL POSITION IN THE LIST OF GOOD UNITS
        UIDs = aid.assign_unique_id(output_prob_matrix, um_param, clus_info)

        with open('UIDs.pkl','wb') as f:
            pickle.dump(UIDs, f)
        if SaveUIDToCSV:
            with open('C:/Users/BizLab/Documents/neuropixels_visualisation/scripts/UIDs_channelcorr_interpolation_UM_drift_corr.csv', mode='w', newline='') as file:
                # Create a csv.writer object
                writer = csv.writer(file)
                # Write data to the CSV file
                writer.writerows(UIDs)
    print('break to get UID')
else:
    with open('UIDs.pkl','rb') as f:
        UIDs= pickle.load(f)

if LoadPickleBombcellAndKilosort:
    with open('Spike_infos_and_cluster_status.pkl','rb') as f:
        [spike_positions_all, spike_clusters_all, spike_times_all, spike_amps_all,
                   nb_units_session, id_kept_Bc, GU_and_MUA, Id_Bc_all_clus]= pickle.load(f)
    spike_pos_by_clus, spike_time_by_clus, spike_amps_by_clus, original_clus_id= create_list_all_units(spike_positions_all, spike_clusters_all, spike_times_all, spike_amps_all)
    Df_clusters= create_dataframe_of_cluster(spike_pos_by_clus, spike_time_by_clus, spike_amps_by_clus, original_clus_id,
                             nb_units_session, motion_sessions, id_kept_Bc)

else:
    spike_positions_all, spike_clusters_all, spike_times_all, spike_amps_all, nb_units_session, id_kept_Bc, GU_and_MUA, Id_Bc_all_clus= load_kilosort_and_bombcell_files(KS_dirs)
    with open('Spike_infos_and_cluster_status.pkl', 'wb') as f:
        pickle.dump([spike_positions_all, spike_clusters_all, spike_times_all, spike_amps_all,
                   nb_units_session, id_kept_Bc, GU_and_MUA, Id_Bc_all_clus], f)
    spike_pos_by_clus, spike_time_by_clus, spike_amps_by_clus, original_clus_id= create_list_all_units(spike_positions_all, spike_clusters_all, spike_times_all, spike_amps_all)
    Df_clusters= create_dataframe_of_cluster(spike_pos_by_clus, spike_time_by_clus, spike_amps_by_clus, original_clus_id,
                             nb_units_session, motion_sessions, id_kept_Bc)
if LoadTDTCorrectedTimes:
    with open("E:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Output/tempDir/F2302_Challah/PFC_shank0_Challah/PFC_shank0/everythingAllAtOnce/corrected_spike_trains.pkl",'rb') as f:
            time_corrected_spiketrains= pickle.load(f)
    #time_corrected_spiketrains= pd.read_csv("E:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Output/tempDir/F2302_Challah/PFC_shank0_Challah/PFC_shank0/everythingAllAtOnce/corrected_spike_trains.csv")
    time_corrected_to_tdt= []
    for i in range(time_corrected_spiketrains.shape[0]):
        clusters_this_session= Df_clusters['Ori ID Bc per session'][Df_clusters['Session']== time_corrected_spiketrains['Session'].iloc[i]].to_list()
        if time_corrected_spiketrains['Cluster ID this session'].iloc[i] in clusters_this_session:
            time_corrected_to_tdt.append(time_corrected_spiketrains['Times'].iloc[i])
    Df_clusters['Time corrected to TDT']= time_corrected_to_tdt

Df_behavior= get_df_behavior(behavior_path, sessionNames)

Df_GU_MUA= filter_GU_MUA(Df_clusters, GU_and_MUA, Id_Bc_all_clus)
Df_Matched_GU_MUA= get_UM_id(Df_GU_MUA, UIDs)
non_matched_units, matched_within, matched_accross, matched_both_ways= check_matching_status(Df_Matched_GU_MUA)
add_matching_status_to_df(Df_Matched_GU_MUA, non_matched_units, matched_within, matched_accross, matched_both_ways)
get_corrected_average_positions(Df_Matched_GU_MUA, motion_sessions)
add_var_pos(Df_Matched_GU_MUA)
add_isi(Df_Matched_GU_MUA)
add_average_firing_rate(Df_Matched_GU_MUA)
get_quantile_amplitude_per_session(Df_Matched_GU_MUA)
Df_Matched_units= Df_Matched_GU_MUA[Df_Matched_GU_MUA['Matching status']=='Accross']

plot_rasters_units(Df_Matched_units, Df_behavior, Trials_alignement_variable= 'lickRelease')

print('Plotting for motion correction without interpolation')
get_df_and_plots(Df_clusters_old, GU_and_MUA, Id_Bc_all_clus, UID_no_interpolation_no_UM_drift)






