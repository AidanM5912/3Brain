import os
import numpy as np
import zipfile
import tempfile
import matplotlib.pyplot as plt
from braingeneers.utils import s3wrangler as wr


class SpikeDataLoader:
    def __init__(self, input_path):
        # Input path can be a string or a list of strings
        if isinstance(input_path, str):
            self.input_paths = [input_path]
        elif isinstance(input_path, list):
            self.input_paths = input_path
        else:
            raise ValueError("input_path must be a string or a list of strings.")
        
        # Initialize lists for multi-dataset support
        self.data_list = []
        self.trains = []
        self.firing_rates_list = []
        self.durations = []
        self.neuron_data_list = []
        self.num_neurons_list = []

        # Load and prepare data for each input path
        for path in self.input_paths:
            self.load_and_prepare_data(path)

        # If only one dataset, set main attributes for single-dataset analyses
        if len(self.input_paths) == 1:
            self.data = self.data_list[0]
            self.train = self.trains[0]
            self.firing_rates = self.firing_rates_list[0]
            self.duration = self.durations[0]
            self.neuron_data = self.neuron_data_list[0]
            self.number_of_neurons = self.num_neurons_list[0]

    def load_and_prepare_data(self, input_path):
        # Check if the file is a .zip containing one .npz
        if zipfile.is_zipfile(input_path):
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(input_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                    
                    # Find the .npz file in the temporary directory
                    npz_files = [f for f in os.listdir(tmp_dir) if f.endswith('.npz')]
                    if len(npz_files) != 1:
                        raise ValueError(f"Expected one .npz file in {input_path}, found {len(npz_files)}.")

                    npz_path = os.path.join(tmp_dir, npz_files[0])
                    self.load_npz_file(npz_path)
        else:
            # If input_path is a single .npz file, load it directly
            self.load_npz_file(input_path)

    def load_npz_file(self, npz_path):
        try:
            # Load the .npz file
            data = np.load(npz_path, allow_pickle=True)
            print(f"File loaded successfully: {npz_path}")
            
            # Validate that the required fields are present
            required_fields = ["train", "fs"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field '{field}' in {npz_path}")

            # Extract spike times and sampling rate
            spike_times = data["train"].item()
            sampling_rate = data["fs"]
            train = [times / sampling_rate for _, times in spike_times.items()]

            # Store data and attributes
            self.data_list.append(data)
            self.trains.append(train)
            self.durations.append(max([max(times) for times in train]))
            firing_rates = [len(neuron_spikes) / max([max(times) for times in train]) for neuron_spikes in train]
            self.firing_rates_list.append(firing_rates)
            self.neuron_data_list.append(data.get("neuron_data", {}).item())
            self.num_neurons_list.append(len(train))
        except Exception as e:
            print(f"Error loading .npz file {npz_path}: {e}")
            raise


def download_from_s3(s3_path, local_path):
    """Download a file from S3 to a local path."""
    print(f"Downloading {s3_path} to {local_path}")
    try:
        wr.download(s3_path, local_path)
    except Exception as e:
        print(f"Error downloading {s3_path}: {e}")
        raise


def get_population_fr(train, bin_size=0.1, w=5):
    """Calculate smoothed population firing rate."""
    trains = np.hstack(train)
    rec_length = np.max(trains)
    bin_num = int(rec_length // bin_size) + 1
    bins = np.linspace(0, rec_length, bin_num)
    fr = np.histogram(trains, bins)[0] / bin_size
    fr_avg = np.convolve(fr, np.ones(w), 'same') / w
    return bins[1:], fr_avg


def plot_high_activity_rasters(train, duration, output_path, dataset_name):
    """Generate smaller raster plots for high-activity periods."""
    print(f"Creating high-activity raster plots for dataset {dataset_name}")
    os.makedirs(output_path, exist_ok=True)

    # Calculate population firing rate
    bins, fr_avg = get_population_fr(train)

    if len(fr_avg) == 0 or np.all(fr_avg == 0):
        print(f"No activity detected in dataset {dataset_name}. Skipping high-activity rasters.")
        return

    # Identify high-activity periods
    threshold = np.percentile(fr_avg, 90)
    high_activity_times = bins[np.where(fr_avg > threshold)]

    if len(high_activity_times) == 0:
        print(f"No high-activity periods found for dataset {dataset_name}.")
        return

    # Define the smaller raster window sizes
    window_sizes = [60, 30, 10]

    for window_size in window_sizes:
        for center_time in high_activity_times:
            start_time = max(0, center_time - window_size / 2)
            end_time = start_time + window_size

            # Generate raster plot
            fig, ax = plt.subplots(figsize=(12, 8))
            y = 0
            for vv in train:
                spikes_in_window = vv[(vv >= start_time) & (vv <= end_time)]
                ax.scatter(spikes_in_window, [y] * len(spikes_in_window), marker="|", c='k', s=4, alpha=0.7)
                y += 1

            # Overlay population firing rate
            bins_in_window = (bins >= start_time) & (bins <= end_time)
            ax2 = ax.twinx()
            ax2.plot(bins[bins_in_window], fr_avg[bins_in_window], color='red', alpha=0.5, linewidth=2)
            ax2.set_ylabel("Population Firing Rate (Hz)", color='red')

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Neuron Index")
            ax.set_xlim(start_time, end_time)
            ax.set_title(f"Raster Plot ({window_size}s) - High Activity: {dataset_name}")

            plot_filename = os.path.join(output_path, f"raster_{dataset_name}_window_{window_size}s_{int(center_time)}s.png")
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close(fig)




def main():
    # Define the S3 paths and dataset names
    datasets = {
        "150uM": "s3://braingeneers/ephys/2024-11-07-e-SmitsMidbrain-24hour/derived/kilosort2/24432_SmitsMO_D55_sect300_T2PostDrug150-24hr_20241107_acqm.zip",
        "Control": "s3://braingeneers/ephys/2024-11-07-e-SmitsMidbrain-24hour/derived/kilosort2/24481_SmitsMO_D55_sect300_Control-24hr_20241107_acqm.zip",
        "175uM": "s3://braingeneers/ephys/2024-11-07-e-SmitsMidbrain-24hour/derived/kilosort2/24578_SmitsMO_D55_sect300_T2PostDrug175-24hr_20241107_acqm.zip"
    }

    output_folder = "high_activity_rasters"
    os.makedirs(output_folder, exist_ok=True)

    for name, s3_path in datasets.items():
        # Prepare local paths
        local_file = os.path.join(output_folder, f"{name}.zip")
        # Download data
        download_from_s3(s3_path, local_file)
        
        # Load spike train data
        loader = SpikeDataLoader(local_file)
        train, duration = loader.train, loader.duration
        
        # Generate high-activity raster plots
        plot_high_activity_rasters(train, duration, output_folder, name)


if __name__ == "__main__":
    main()
