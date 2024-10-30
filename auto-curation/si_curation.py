# curate by spikeinterface quality metrics and curation
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.core as sc
import spikeinterface.curation as curation
import spikeinterface.qualitymetrics as sqm
from spikeinterface.extractors.neoextractors import MaxwellRecordingExtractor
import spikeinterface.preprocessing as spre
import argparse
import sys
import posixpath
import os
import shutil
import braingeneers.utils.s3wrangler as wr
from utils import *
import logging
import h5py
import json

# BUCKET = "s3://braingeneers/ephys/"
JOB_KWARGS = dict(n_jobs=10, progress_bar=True)
os.environ["HDF5_PLUGIN_PATH"] = os.getcwd()
LOG_FILE_NAME = "run_autocuration.log"
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_NAME, mode="a"),
                              stream_handler])

DEFUALT_PARAMS = {"min_snr": 5,
                  "min_fr": 0.5,
                  "max_isi_viol": 1}

class QualityMetrics:
    """
    curation by quality metrics using spikeinterface API

    """

    def __init__(self, base_folder, rec_path, phy_folder, 
                 data_format=None, params_dict={},
                 max_spikes_waveform=500,
                 default=True):
        self.redundant_pairs = None
        self.extract_path = None
        self.rec_path = rec_path
        self.base_folder = base_folder
        self.data_format = data_format
        self.clean_folder = posixpath.join(base_folder, "cleaned_waveforms")

        phy_result = se.KiloSortSortingExtractor(phy_folder)
        self.phy_result = phy_result.remove_empty_units()
        if "min_snr" in params_dict:
            self._snr_thres = params_dict["min_snr"]
        else:
            self._snr_thres = DEFUALT_PARAMS["min_snr"]
        if "min_fr" in params_dict:
            self._fr_thres = params_dict["min_fr"]
        else:
            self._fr_thres = DEFUALT_PARAMS["min_fr"]
        if "max_isi_viol" in params_dict:
            self._isi_viol_thres = params_dict["max_isi_viol"]
        else:
            self._isi_viol_thres = DEFUALT_PARAMS["max_isi_viol"]
        
        # extract waveforms
        self.we = self.extract_waveforms(max_spikes=max_spikes_waveform)
        print("waveforms", self.we)
        if default:  # to leave space for other curation methods
            self.curated_ids, self.all_remove_ids = self.default_curation()

        logging.info("Saving cleaned units...")
        self.we_clean = self.we.select_units(self.curated_ids, self.clean_folder)
        print("Saved ", self.we_clean)

    def default_curation(self):
        all_remove_ids = set()
        
        # Step-by-step logging of removal from each curation step
        ids_snr = self.curate_by_snr
        logging.info(f"Units removed by SNR curation: {ids_snr}")
        all_remove_ids.update(ids_snr)

        ids_isi = self.curate_by_isi()
        logging.info(f"Units removed by ISI curation: {ids_isi}")
        all_remove_ids.update(ids_isi)

        ids_fr = self.curate_by_fr()
        logging.info(f"Units removed by firing rate curation: {ids_fr}")
        all_remove_ids.update(ids_fr)

        # Log cumulative removals after all steps
        self.redundant_pairs = self.curate_by_redundant()
        logging.info(f"Total unique units flagged for removal across all steps: {len(all_remove_ids)}")
        
        # Log curation completion and unique list of units for removal
        curated_excess = curation.remove_excess_spikes(self.we.sorting, self.we.recording)
        self.we.sorting = curated_excess
        logging.info(f"Final curated units (excess spikes removed): {curated_excess.unit_ids}")
        
        return curated_excess.unit_ids, list(all_remove_ids)

    def prepare_rec(self, low=300., high=6000., common_ref=True):
        if self.data_format == "Maxwell":
            rec = MaxwellRecordingExtractor(file_path=self.rec_path)
            gain_uv = read_maxwell_gain(self.rec_path)
        elif self.data_format == "nwb":
            rec = se.read_nwb(self.rec_path)
            gain_uv = 1
        rec_scale = spre.ScaleRecording(rec, gain=gain_uv)
        rec_filt = spre.bandpass_filter(rec_scale, freq_min=low, freq_max=high, dtype="float32")
        if common_ref:
            rec_cmr = spre.common_reference(rec_filt)
            return rec_cmr
        else:
            return rec_filt

    def extract_waveforms(self, ms_before=2., ms_after=3., max_spikes=500):
        rec_pre = self.prepare_rec()
        self.extract_path = posixpath.join(self.base_folder, "extract_waveforms")
        
        if os.path.isdir(self.extract_path):
            logging.info("Loading WaveformExtractor from existing directory.")
            we = sc.WaveformExtractor.load(folder=self.extract_path)
            logging.info("WaveformExtractor loaded from saved directory.")
        else:
            logging.info("Creating new WaveformExtractor.")
            we = sc.WaveformExtractor(rec_pre, self.phy_result, self.base_folder, allow_unfiltered=False)
            we.set_params(ms_before=ms_before, ms_after=ms_after, max_spikes_per_unit=max_spikes)
            logging.info("Parameters set for WaveformExtractor.")
            we.run_extract_waveforms(**JOB_KWARGS)
            logging.info("Waveform extraction run completed.")
            we.save(self.extract_path, overwrite=True)
            logging.info(f"WaveformExtractor saved to path: {self.extract_path}")
        
        # Verify structure of the newly created or loaded WaveformExtractor
        logging.info(f"Type of we: {type(we)}")
        logging.info(f"Unit IDs in we: {we.unit_ids}")
        logging.info(f"Is sorting set in we: {we.sorting is not None}")
        
        try:
            templates_preview = we.get_all_templates()[:5]  # Preview first few templates
            logging.info(f"Extracted templates (preview): {templates_preview}")
        except Exception as e:
            logging.error(f"Error when accessing templates in we: {e}")
        
        self.we_clean = we  # Assign to self.we_clean for accessibility in other functions
        return we


    def compute_noise_level(self):
        rec_pre = self.prepare_rec()
        noise_levels_mv = si.get_noise_levels(rec_pre, return_scaled=True)
        return noise_levels_mv

    

    @property
    def curate_by_snr(self):
        num_units = len(self.we.unit_ids)
        snr = sqm.compute_snrs(self.we)
        remove_ids = []
        for k, v in snr.items():
            if v < self._snr_thres:
                remove_ids.append(k)
        cleaned_sorting = self.we.sorting.remove_units(remove_ids)
        self.we.sorting = cleaned_sorting
        logging.info(f"Curated by SNR of {self._snr_thres} rms. "
                     f"Remove number of units: {len(remove_ids)}/{num_units}")
        return remove_ids

    def curate_by_isi(self):
        """
        ISI violation by Hill method with 1.5 ms refactory period
        """
        num_units = len(self.we.unit_ids)
        isi_viol_ratio, isi_viol_num = sqm.compute_isi_violations(self.we)
        remove_ids = []
        for k, v in isi_viol_ratio.items():
            if v > self._isi_viol_thres:
                remove_ids.append(k)
        cleaned_sorting = self.we.sorting.remove_units(remove_ids)
        self.we.sorting = cleaned_sorting
        logging.info(f"Curated by ISI violation (Hill method) "
                     f"of {self._isi_viol_thres}/1 of 1.5 ms refactory period. "
                     f"Remove number of units: {len(remove_ids)}/{num_units}")
        return remove_ids
    
    def curate_by_isi_ratio(self):
        """
        ISI violation by ratio defined as number of violations over total number of spikes
        """
        # TODO
        
        pass

    def curate_by_fr(self):
        num_units = len(self.we.unit_ids)
        firing_rate = sqm.compute_firing_rates(self.we)
        remove_ids = []
        for k, v in firing_rate.items():
            if v < self._fr_thres:
                remove_ids.append(k)
        cleaned_sorting = self.we.sorting.remove_units(remove_ids)
        self.we.sorting = cleaned_sorting
        logging.info(f"Curated by firing rate of {self._fr_thres} Hz. "
                     f"Remove number of units: {len(remove_ids)}/{num_units}")
        return remove_ids

    def curate_by_redundant(self):
        num_units = len(self.we.unit_ids)
        curated_redundant, redundant_unit_pairs = \
            curation.remove_redundant_units(self.we, align=False,
                                            remove_strategy="max_spikes", extra_outputs=True)
        print("done redundant")

        remove_ids = np.setdiff1d(self.we.sorting.unit_ids, curated_redundant.unit_ids)
        logging.info(f"Curated by checking redundant units (Function turned off, no unit removed). "
                     f"Found number of units to remove: {len(remove_ids)}/{num_units}")
        # self.we.sorting = curated_redundant
        # return remove_ids
        return redundant_unit_pairs

    def package_cleaned(self):
        logging.info("Verifying self.we_clean structure before packaging.")
        logging.info(f"Unit IDs: {self.we_clean.unit_ids if self.we_clean else 'we_clean not initialized'}")
        
        try:
            spike_data = self.compile_data()
            logging.info(f"Compiled spike data successfully: {list(spike_data.keys())}")
        except Exception as e:
            logging.error(f"Error compiling spike data: {e}")
            raise

        curated_file = 'qm.npz'
        curated_folder = posixpath.join(self.base_folder, "curated")
        if not os.path.isdir(curated_folder):
            os.mkdir(curated_folder)

        qm_npz = posixpath.join(curated_folder, curated_file)

        try:
            np.savez(qm_npz, **spike_data)
            logging.info(f"Saved spike data to {qm_npz}")
        except Exception as e:
            logging.error(f"Error saving spike data: {e}")
            raise

        if os.path.exists(LOG_FILE_NAME):
            logging.info(f"Moving log file {LOG_FILE_NAME} to {curated_folder}")
            shutil.move(LOG_FILE_NAME, curated_folder)
        else:
            logging.error(f"Log file {LOG_FILE_NAME} not found.")
        
        logging.info(f"Contents of curated_folder before zipping: {os.listdir(curated_folder)}")
        try:
            qm_file = shutil.make_archive(posixpath.join(self.base_folder, "qm"), format="zip", root_dir=curated_folder)
            logging.info(f"Created qm_file at {qm_file}")
        except Exception as e:
            logging.error(f"Error creating qm_file archive: {e}")
            raise

        # also package waveforms
        rec_attr = posixpath.join(self.extract_path, "recording_info", "recording_attributes.json")
        if os.path.isfile(rec_attr):
            shutil.copy(rec_attr, self.clean_folder)
        else: 
            logging.warning(f"Recording attributes {rec_attr} not found in extract path.")

        try:
            wf_file = shutil.make_archive(posixpath.join(self.base_folder, "wf"), format="zip", root_dir=self.clean_folder)
            logging.info(f"Created wf_file at {wf_file}")
        except Exception as e:
            logging.error(f"Error creating wf_file archive: {e}")
            raise
        
        return qm_file, wf_file
    
    def compile_data(self, n=12):
        """
        Compile the cleaned sorting to npz with braingeneers-compatible structure.
        """
        logging.info("Starting compile_data...")

        # Retrieve templates
        try:
            templates = self.we_clean.get_all_templates()
            logging.info("Templates extracted successfully.")
        except Exception as e:
            logging.error(f"Error extracting templates: {e}")
            raise

        # Retrieve clusters
        clusters = self.we_clean.unit_ids
        logging.info(f"Number of clusters (units): {len(clusters)}")
        
        # Retrieve channels and positions
        try:
            channels = self.we_clean.recording.get_channel_ids()
            positions = self.we_clean.recording.get_channel_locations()
            logging.info(f"Recording channels count: {len(channels)}; Positions shape: {positions.shape}")
        except Exception as e:
            logging.error(f"Error retrieving channels or positions: {e}")
            raise
        
        # Determine best channels
        try:
            best_channels = get_best_channel_cluster(clusters, channels, templates)
            logging.info(f"Best channels determined successfully. Sample: {list(best_channels.items())[:5]}")
        except Exception as e:
            logging.error(f"Error determining best channels: {e}")
            raise

        # Populate neuron_dict
        neuron_dict = dict.fromkeys(np.arange(len(clusters)), None)
        for i, c in enumerate(clusters):
            try:
                temp = templates[i]
                sorted_idx = sort_template_amplitude(temp)[:n]
                temp = temp.T
                best_idx = sorted_idx[0]

                neuron_dict[i] = {
                    "cluster_id": c,
                    "channel": best_channels[c],
                    "position": positions[best_idx],
                    "template": temp[best_idx],
                    "neighbor_channels": channels[sorted_idx],
                    "neighbor_positions": positions[sorted_idx],
                    "neighbor_templates": temp[sorted_idx]
                }
                logging.info(f"Cluster {c} populated in neuron_dict.")
            except Exception as e:
                logging.error(f"Error processing cluster {c}: {e}")
                raise

        # Check data format and configuration
        if self.data_format == "Maxwell":
            try:
                config = read_maxwell_mapping(self.rec_path)
                logging.info("Electrode configuration read for Maxwell format.")
            except Exception as e:
                logging.error(f"Error reading Maxwell configuration: {e}")
                config = {}
        else:
            logging.info("No electrode configuration needed for non-Maxwell format.")
            config = {}

        # Final compilation of spike data
        try:
            spike_data = {
                "train": {c: self.we_clean.sorting.get_unit_spike_train(c) for c in clusters},
                "neuron_data": neuron_dict,
                "config": config,
                "redundant_pairs": self.redundant_pairs,
                "fs": self.we_clean.recording.sampling_frequency
            }
            logging.info("Spike data compiled successfully.")
        except Exception as e:
            logging.error(f"Error compiling spike data: {e}")
            raise

        return spike_data


def read_maxwell_gain(h5_file):
    dataset = h5py.File(h5_file, 'r')
    if 'mapping' in dataset.keys():
        gain_uv = dataset['settings']['lsb'][0] * 1e6
    else:
        gain_uv = dataset['recordings']['rec0000']['well000']['settings']['lsb'][0] * 1e6
    return gain_uv


def read_maxwell_mapping(h5_file):
    with h5py.File(h5_file, 'r') as dataset:
        if 'version' and 'mxw_version' in dataset.keys():
            mapping = dataset['recordings']['rec0000']['well000']['settings']['mapping']
            config = {'pos_x': np.array(mapping['x']),
                      'pos_y': np.array(mapping['y']),
                      'channel': np.array(mapping['channel']),
                      'electrode': np.array(mapping['electrode'])}
        else:
            mapping = dataset['mapping']
            config = {'pos_x': np.array(mapping['x']),
                      'pos_y': np.array(mapping['y']),
                      'channel': np.array(mapping['channel']),
                      'electrode': np.array(mapping['electrode'])}
    return config


def get_parent_data(neuron_dict):
    parent_id_dict = {v["cluster_id"]: v for _, v in neuron_dict.items()}
    parent_ids = list(parent_id_dict.keys())
    return parent_ids, parent_id_dict


def select_units(spike_train, neuron_dict, selected_ids):
    parent_ids, parent_dict = get_parent_data(neuron_dict)
    update_dict = {}
    update_trains = []
    for i in range(len(selected_ids)):
        id = selected_ids[i]
        update_dict[i] = parent_dict[id]
        update_trains.append(spike_train[parent_ids.index(id)])
    return update_trains, update_dict


def remove_units(spike_train, neuron_dict, removed_ids):
    parent_ids, _ = get_parent_data(neuron_dict)
    selected_ids = np.setdiff1d(parent_ids, removed_ids)
    update_trains, update_dict = select_units(spike_train, neuron_dict, selected_ids)
    return update_trains, update_dict

def upload_file(phy_path, local_file, params_file_name=None, file_type="qm"):
    # Adjust the upload path based on the file type
    if params_file_name is None:
        upload_suffix = f"_ac{file_type}.zip"  # Allows `_acqm.zip` and `_acwf.zip`
    else:
        upload_suffix = f"_{params_file_name}_ac{file_type}.zip"
    
    upload_path = phy_path.replace("_phy.zip", upload_suffix)
    
    # Log and upload
    logging.info(f"Uploading data from {local_file} to {upload_path} ...")
    logging.info(f"Final upload path: {upload_path}")
    
    if not os.path.exists(local_file):
        logging.error(f"Local file {local_file} does not exist. Cannot proceed with upload.")
        return
    
    wr.upload(local_file=local_file, path=upload_path)
    logging.info("Done!")


def parse_uuid(data_path):
    # Remove any trailing slashes
    data_path = data_path.rstrip("/")
    
    # Extract 'experiment' as the last part of the path and define 'base_path'
    experiment = data_path.split("/")[-3]
    base_path = data_path.rsplit("/", 2)[0]  # Get up to 'Ventral_3DAccura_00'
    
    # Define paths based on the 'original/data' or 'shared' in base_path
    if "original/data" in data_path:
        phy_base_path = posixpath.join(base_path, "derived/kilosort2")
        metadata_path = posixpath.join(base_path, "metadata.json")
    elif "shared" in data_path:
        phy_base_path = posixpath.join(base_path, "derived/kilosort2")
        metadata_path = posixpath.join(base_path, "metadata.json")
    else:
        # If neither "original/data" nor "shared" are found, set default paths
        phy_base_path = posixpath.join(base_path, "derived/kilosort2")
        metadata_path = posixpath.join(base_path, "metadata.json")

    # Clean up 'experiment' to remove file extensions
    if experiment.endswith(".raw.h5") or experiment.endswith(".h5"):
        experiment = experiment.rsplit(".", 1)[0]
    elif experiment.endswith(".brw") or experiment.endswith(".nwb"):
        experiment = experiment.rsplit(".", 1)[0]

    # Define 'phy_path' based on 'phy_base_path' and 'experiment'
    phy_path = posixpath.join(phy_base_path, experiment + "_phy.zip")
    return base_path, experiment, metadata_path, phy_path

def hash_file_name(input_string):
    import hashlib
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    hash_string = md5_hash.hexdigest()
    return hash_string

if __name__ == "__main__":
    # test data: s3://braingeneers/ephys/2024-01-05-e-uploader-test/original/data/test_0.raw.h5 
    # test parameter: s3://braingeneers/services/mqtt_job_listener/params/curation/params_1.json

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run curation script with optional parameters.")
    parser.add_argument("data_path", type=str, help="S3 path to the data file.")
    parser.add_argument("param_path", type=str, nargs="?", default=None, help="Optional S3 path to the parameter file.")
    
    # Parse arguments
    args = parser.parse_args()
    data_path = args.data_path
    param_path = args.param_path
    
    # Use param_path or fall back to default
    params_file_name = param_path.split("/")[-1].split(".")[0] if param_path else "params_default"

    s3_base_path, experiment, metadata_path, phy_path = parse_uuid(data_path=data_path)
    print(f"s3 path: {data_path}")  # original recording s3 full path
    print(f"s3 base: {s3_base_path}")
    print(f"metadata path: {metadata_path}")
    print(f"phy path: {phy_path}")
    print(f"parameter file path: {param_path}")


    # Setup paths for download and extraction
    current_folder = os.getcwd()
    base_folder = os.path.join(current_folder, "data")
    os.makedirs(base_folder, exist_ok=True)
    extract_dir = os.path.join(base_folder, "kilosort_result")
    kilosort_local_path = os.path.join(base_folder, "kilosort_result.zip")
    metadata_local_path = os.path.join(base_folder, "metadata.json")


###remove, gives annoying errors
    #for p in [phy_path, data_path, param_path]:
    #    try:
    #        assert wr.does_object_exist(p)
    #    except AssertionError as err:
    #        logging.exception(f"File doesn't exist on S3! {p}")
    #        logging.info("Program exited")
    #        raise err
    
    # download metadata
    if wr.does_object_exist(metadata_path):
        logging.info("Start downloading metadata ...")
        wr.download(metadata_path, metadata_local_path)
        logging.info("Done!")
        with open(metadata_local_path, "r") as f:
            metadata = json.load(f)
            if (experiment in metadata["ephys_experiments"]) and \
                    ("data_format" in metadata["ephys_experiments"][experiment]):
                data_format = metadata["ephys_experiments"][experiment]["data_format"]
                logging.info(f"Read data format from metadata.json, format is {data_format}")
            else:
                data_format = "Maxwell"  # a patch for the old metadata.json
                logging.info(f"Data format not found in metadata.json, default to Maxwell")
    else:
        logging.info("Metadata file not found. Skip downloading metadata.")
        logging.info("Data format default to Maxwell")
        data_format = "Maxwell"

    # download phy.zip
    logging.info("Start downloading kilosort result ...")
    wr.download(phy_path, kilosort_local_path)
    logging.info("Done!")
    shutil.unpack_archive(kilosort_local_path, extract_dir, "zip")

    # Define and create shared directory path
    nwb_s3_path = f"{s3_base_path}/shared/{experiment}.nwb"
    raw_data_local_path = posixpath.join(base_folder, "shared", f"{experiment}.nwb")
    os.makedirs(os.path.join(base_folder, "shared"), exist_ok=True)  # Ensure directory exists

    # Download the NWB file for raw data
    logging.info(f"Start downloading raw data from {nwb_s3_path} ...")
    wr.download(nwb_s3_path, raw_data_local_path)
    logging.info("Raw data download complete!")



    ## download raw data (getting rid of this for now)
    ##come back and debug if need to recurate for different files
    #logging.info("Start downloading raw data ...")
    #experiment = "rec"
    #wr.download(data_path, posixpath.join(base_folder, experiment))
    #logging.info("Done")


    # download param file
    # Download and load parameter file if param_path is provided
    if param_path:
        logging.info("Start downloading parameter file ...")
        param_file = posixpath.join(base_folder, "params.json")
        wr.download(param_path, param_file)
        logging.info("Done")
        with open(param_file, "r") as f:
            params_dict = json.load(f)

        if len(params_dict) > 0:
            logging.info(f"Use parameters {params_dict} from file {param_path} for curation")
        else:
            params_dict = DEFUALT_PARAMS
            params_file_name = "params_default"
            logging.info(f"User parameters not available. Using default parameters {DEFUALT_PARAMS} for curation")
    else:
        # Fallback to default parameters if no param_path is provided
        params_dict = DEFUALT_PARAMS
        params_file_name = "params_default"
        logging.info(f"No parameter file provided. Using default parameters: {params_dict}")


    # do curation
    curation = QualityMetrics(base_folder=base_folder, 
                              rec_path=raw_data_local_path, 
                              phy_folder=extract_dir,
                              data_format=data_format, 
                              params_dict=params_dict)
    qm_file, wf_file = curation.package_cleaned()

    # curated_file = experiment + "_qm.zip"
    # waveform_file = experiment + "_wf.zip"
    upload_file(phy_path, qm_file, params_file_name, file_type="qm")
    upload_file(phy_path, wf_file, params_file_name, file_type="wf")
    # upload_file(uuid, wf_file, waveform_file)
