import os
import numpy as np
import zipfile
import tempfile
import matplotlib.pyplot as plt
from braingeneers.utils import s3wrangler as wr


class SpikeDataAnalysis:
    def __init__(self, input_path, original_paths=None):
        # input_path to be a single string or a list of strings
        if isinstance(input_path, str):
            self.input_paths = [input_path]
        elif isinstance(input_path, list):
            self.input_paths = input_path
        else:
            raise ValueError("input_path must be a string or a list of strings.")
        
        #reference_paths needed for zip file upload naming
        # use original_paths for naming if provided; otherwise fall back to local_paths
        reference_paths = original_paths if original_paths is not None else self.input_paths

        #generate original and cleaned names
        self.original_names = [os.path.splitext(os.path.basename(path))[0] for path in reference_paths]
        self.cleaned_names = [self.clean_name(name, self.original_names) for name in self.original_names]

        # initialize lists for multi-dataset support
        self.data_list = []
        self.trains = []
        self.firing_rates_list = []
        self.durations = []
        self.neuron_data_list = []
        self.num_neurons_list = []

        # load and prepare data for each input path
        for path in self.input_paths:
            self.load_and_prepare_data(path)

        # if only one dataset, set main attributes for single-dataset analyses
        if len(self.input_paths) == 1:
            self.data = self.data_list[0]
            self.train = self.trains[0]
            self.firing_rates = self.firing_rates_list[0]
            self.duration = self.durations[0]
            self.neuron_data = self.neuron_data_list[0]
            self.number_of_neurons = self.num_neurons_list[0]

    def load_and_prepare_data(self, input_path):
        # Check if the file is a .zip containing one .npz
        print(f"Processing input_path: {input_path}")  # debugging
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
            # load the .npz file
            data = np.load(npz_path, allow_pickle=True)
            print(f"File loaded successfully: {npz_path}")
            
            # validate that the required fields are present
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

        except FileNotFoundError:
            print(f"Error: File {npz_path} not found.")
            raise
        except ValueError as e:
            print(f"Error in file {npz_path}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error loading file {npz_path}: {e}")
            raise

    def clean_name(self, base_name, all_base_names):

        # Normalize to lowercase for case-insensitive comparison
        normalized_names = [re.sub(r'\W+', '_', name.lower()) for name in all_base_names]  # Replace non-alphanumeric with "_"
        base_name_normalized = re.sub(r'\W+', '_', base_name.lower())

        # Split names into components
        split_names = [set(name.split("_")) for name in normalized_names]
        common_parts = set.intersection(*split_names)  # Find common components across all names

        # Remove common parts and rejoin
        cleaned_parts = [part for part in base_name_normalized.split("_") if part not in common_parts]
        cleaned_name = "_".join(cleaned_parts)

        # Return the original name if cleaning results in an empty string
        return cleaned_name if cleaned_name else base_name




    @staticmethod
    def get_population_fr(train, bin_size=0.1, sigma=5, smoothing=True):
        """
        Compute the population firing rate from spike trains. Smoothed with a gaussian filter.

        ***
        Used to have an option to normalize by the number of neurons. Option has since been removed.
        Now we always normalize by number of neurons to get Hz/Neuron.

        There is a chance of neuron death affecting the number of neurons, and hence the firing rate.
        This may be smoothed over because by this normaliziation. If this is a concern, consider multiplying
        by the number of neurons 
        ***

        Parameters:
            train (list of arrays): Spike trains, each array corresponds to a neuron's spike times.
            bin_size (float): Size of the time bins in seconds.
            sigma (float): Standard deviation of the Gaussian filter.
            average (bool): If True, normalize the firing rate by the number of neurons.
        """
        from scipy.ndimage import gaussian_filter1d
        trains = np.hstack(train)
        rec_length = np.max(trains)
        bin_num = int(rec_length // bin_size) + 1
        bins = np.linspace(0, rec_length, bin_num)
        fr = np.histogram(trains, bins)[0] / bin_size

        num_neurons = len(train)  
        fr_normalized = fr / num_neurons  # firing rate normalized to Hz/Neuron

        if smoothing:
            fr_smoothed = gaussian_filter1d(fr_normalized, sigma=sigma) #smoothing over bin
            return bins[1:], fr_smoothed  # bin centers and smoothed firing rate
        else: 
            return bins[1:], fr_normalized #bin centers and normalizied firing rate



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




    def raster_plot(self, output_path, dataset_name):
        print(f"Plotting raster plot for {dataset_name}")
        for i, train in enumerate(self.trains):
            fig, ax = plt.subplots(figsize=(10, 8))
            y = 0
            for vv in train:
                plt.scatter(vv, [y] * len(vv), marker="|", c='k', s=4, alpha=0.7)
                y += 1

            num_neurons = len(train)
            tick_spacing = 1 if num_neurons <= 50 else 2 if num_neurons <= 100 else 3 if num_neurons <= 150 else 5
            ax.set_yticks(range(1, num_neurons + 1, tick_spacing))

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron Index')
            ax.set_xlim(0, self.durations[i])

            secax = ax.secondary_xaxis('top')
            secax.set_xlabel("Time (Hours)")
            xticks = ax.get_xticks()
            secax.set_xticks(xticks)
            secax.set_xticklabels([f"{x / 3600:.2f}" for x in xticks])

            ax.set_title(f"Raster Plot: {dataset_name}")
            plt.savefig(os.path.join(output_path, f"raster_{dataset_name}.png"))
            plt.close(fig)

    def footprint_overlay_fr(self, output_path, dataset_name):
        print(f"Plotting footprint overlay with firing rates for {dataset_name}")
        plt.close('all')
        for i, neuron_data in enumerate(self.neuron_data_list):

            neuron_x, neuron_y, filtered_firing_rates = [], [], []

            for j, neuron in enumerate(neuron_data.values()):
                x, y = neuron['position']
                if (x, y) != (0, 0):  # exclude the calibration electrode
                    neuron_x.append(x)
                    neuron_y.append(y)
                    filtered_firing_rates.append(self.firing_rates_list[i][j] / len(self.firing_rates_list[i]))

            filtered_firing_rates = np.array(filtered_firing_rates)
            legend_rates = np.percentile(filtered_firing_rates, [50, 75, 90, 98])

            plt.figure(figsize=(11, 9))
            plt.scatter(neuron_x, neuron_y, s=filtered_firing_rates * 100, alpha=0.4, c='r', edgecolors='none')
        
            for rate in legend_rates:
                plt.scatter([], [], s=rate * 100, c='r', alpha=0.4, label=f'{rate:.2f} kHz')

            plt.legend(scatterpoints=1, frameon=True, labelspacing=1.4, handletextpad=0.8, borderpad=0.92, title='Firing Rate', loc = 'best', title_fontsize=10, fontsize=10)
            plt.xlabel(r"Horizontal Position ($\mu$m)")
            plt.ylabel(r"Vertical Position ($\mu$m)")
            plt.title(f"Footprint Plot with Firing Rates: {dataset_name}")
            plt.savefig(os.path.join(output_path, f"footprint_plot_fr_{dataset_name}.png"))
            plt.close()

    def run_all_analyses(self, output_folder, base_names, perform_pca=False, cleanup=True):
        """
        Execute all analyses for individual datasets and comparisons.
        Includes optional PCA analysis if `perform_pca` is True.
        """
        os.makedirs(output_folder, exist_ok=True)

        dataset_directories = []
        for base_name in base_names:
            dataset_dir = os.path.join(output_folder, base_name)
            os.makedirs(dataset_dir, exist_ok=True)

            # now creating a rasters subdirectory for each dataset. too many small rasters in the main directory seems annoying
            #rasters_dir = os.path.join(dataset_dir, "rasters")
            #os.makedirs(rasters_dir, exist_ok=True)


            dataset_directories.append((dataset_dir, base_name))

        # create a directory for comparison plots
        comparison_dir = os.path.join(output_folder, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)

        #create subdirectories inside comparison to avoid crowding with plots of similar names 
        pairwise_dir = os.path.join(comparison_dir, "linear_comparisons", "pairwise")
        global_dir = os.path.join(comparison_dir, "linear_comparisons", "global")
        os.makedirs(pairwise_dir, exist_ok=True)
        os.makedirs(global_dir, exist_ok=True)

        # perform analyses for each dataset
        for i, (dataset_dir, dataset_name) in enumerate(dataset_directories):
            
            #create rasters 
            #self.raster_plot(rasters_dir, dataset_name)
            #self.plot_high_activity_rasters(rasters_dir, dataset_name)

            #regular analysis
            self.raster_plot(dataset_dir, dataset_name)

        # zip output folder
        zip_filename = f"{output_folder}.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            for foldername, subfolders, filenames in os.walk(output_folder):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    zip_file.write(file_path, os.path.relpath(file_path, output_folder))

        # clean up (optional)
        if cleanup:
            shutil.rmtree(output_folder)

        return zip_filename



def main():
    parser = argparse.ArgumentParser(description="General analysis script for multi-condition neural data.")
    #input s3 paths argument
    parser.add_argument("input_s3", nargs='+', help="Input file paths (S3 paths)")
    #output s3 paths argument
    parser.add_argument("output_path", help="Output S3 path for the zip file")
    #optional flag for pca analysis
    parser.add_argument("--pca", action="store_true", help="Perform PCA analysis on firing rates")
    #optional flag for cleanup
    parser.add_argument("--cleanup", action="store_true", help="Delete the output folder after zipping")
    args = parser.parse_args()

    #access parsed arguments
    input_s3 = args.input_s3
    output_s3 = args.output_path
    perform_pca = args.pca
    cleanup = args.cleanup

    # functionality to download multiple s3 files
    local_input_paths = []
    for i, s3_path in enumerate(input_s3):
        local_file = f'/tmp/input_path_{i}.npz'
        print(f"Downloading {s3_path} to {local_file}")
        try:
            wr.download(s3_path, local_file)
        except Exception as e:
            print(f"Error downloading {s3_path}: {e}")
            sys.exit(1)
        local_input_paths.append(local_file)

    # run analysis
    analysis = SpikeDataAnalysis(local_input_paths, original_paths=input_s3)
    combined_name = "_".join(analysis.cleaned_names[:2])  # use original (cleaned) names for combined_name

    # create local temporary directories for processing
    local_output_folder = f'/tmp/output_plots_{combined_name}'


    zip_filename = analysis.run_all_analyses(local_output_folder, analysis.cleaned_names, perform_pca=perform_pca, cleanup=cleanup)

    output_s3 = os.path.join(output_s3, os.path.basename(zip_filename))
    # upload zip to S3
    print(f"Uploading {zip_filename} to {output_s3}")
    wr.upload(zip_filename, output_s3)

    print("Analysis complete. Results uploaded to S3.")




if __name__ == "__main__":
    main()
