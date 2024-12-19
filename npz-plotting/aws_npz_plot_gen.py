import argparse
import braingeneers.utils.s3wrangler as wr
from datetime import datetime
import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import scipy.stats
import scipy.ndimage
import sklearn.decomposition
import sys
import tempfile
import uuid
import zipfile




class SpikeDataAnalysis:
    def __init__(self, input_path):
        # input_path to be a single string or a list of strings
        if isinstance(input_path, str):
            self.input_paths = [input_path]
        elif isinstance(input_path, list):
            self.input_paths = input_path
        else:
            raise ValueError("input_path must be a string or a list of strings.")
        
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


    def raster_plot(self, output_path, dataset_name):
        print(dataset_name)
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

    def footprint_opaque_circles(self, output_path, dataset_name):
        print(dataset_name)
        plt.close('all')
        for i, neuron_data in enumerate(self.neuron_data_list):

            neuron_x, neuron_y, filtered_firing_rates = [], [], []

            for j, neuron in enumerate(neuron_data.values()):
                x, y = neuron['position']
                if (x, y) != (0, 0):  # exclude the calibration electrode
                    neuron_x.append(x)
                    neuron_y.append(y)
                    filtered_firing_rates.append(self.firing_rates_list[i][j])

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

    @staticmethod
    def get_population_fr(train, bin_size=0.1, sigma=5, average=False):
        from scipy.ndimage import gaussian_filter1d
        trains = np.hstack(train)
        rec_length = np.max(trains)
        bin_num = int(rec_length // bin_size) + 1
        bins = np.linspace(0, rec_length, bin_num)
        fr = np.histogram(trains, bins)[0] / bin_size


        if average:
            # normalize by number of neurons
            num_neurons = len(train)  
            fr_normalized = fr / num_neurons  # firing rate normalized to Hz/Neuron
        else:
            fr_normalized = fr
         
        fr_smoothed = gaussian_filter1d(fr_normalized, sigma=sigma) #smoothing instead of average over bin

        return bins[1:], fr_smoothed  # bin centers and smoothed firing rate


    def plot_smoothed_population_fr(self, output_path, dataset_name):
        plt.close('all')
        for i, train in enumerate(self.trains):

            bins, fr_avg = self.get_population_fr(train, average=False)

            plt.figure(figsize=(12, 6))
            plt.plot(bins, fr_avg)

            plt.xlabel("Time (s)", fontsize=12)
            plt.ylabel("Population Firing Rate (Hz)/Neuron", fontsize=12)

            plt.xlim(0, self.durations[i])
            plt.ylim(np.min(fr_avg) - 5, np.max(fr_avg) + 5)

            plt.title(f"Smoothed  Population Firing Rate: {dataset_name}", fontsize=16)
            plt.savefig(os.path.join(output_path, f"smoothed_population_fr_{dataset_name}.png"))
            plt.close()

    def overlay_fr_raster(self, output_path, dataset_name):
        plt.close('all')
        for i, train in enumerate(self.trains):

            bins, fr_avg = self.get_population_fr(train, average=True)

            fig, axs = plt.subplots(1, 1, figsize=(16, 6))
            axs1 = axs.twinx()

            axs1.plot(bins, fr_avg, color='r', linewidth=3, alpha=0.5)

            y = 0
            for vv in train:
                axs.scatter(vv, [y] * len(vv), marker="|", c='k', s=4, alpha=0.7)
                y += 1

            axs.set_xlabel("Time (s)", fontsize=14)
            axs.set_ylabel("Neuron Number", fontsize=14)
            axs1.set_ylabel("Normalized Population Firing Rate (Hz)", fontsize=16, color='r')
            axs1.spines['right'].set_color('r')
            axs1.spines['right'].set_linewidth(2)
            axs1.tick_params(axis='y', colors='r')

            axs.set_title(f"Population Level Activity: {dataset_name}", fontsize=16)
            plt.savefig(os.path.join(output_path, f"overlay_fr_raster_{dataset_name}.png"))
            plt.close(fig)

    @staticmethod
    def compute_sttc_matrix(spike_train, length, delt=20):

        #handle this case
        if not any(len(ts) > 0 for ts in spike_train):
            print("Warning: Spike train is empty or contains no spikes. Returning zero-filled STTC matrix.")
            return np.zeros((len(spike_train), len(spike_train)))

        def time_in_delt(tA, delt, tmax):
            if len(tA) == 0:
                return 0
            base = min(delt, tA[0]) + min(delt, tmax - tA[-1])
            return base + np.minimum(np.diff(tA), 2 * delt).sum()

        def sttc_pairs(tA, tB, TA, TB, delt):
            def spikes_in_delt(tA, tB, delt):
                if len(tB) == 0:
                    return 0
                tA, tB = np.asarray(tA), np.asarray(tB)
                iB = np.searchsorted(tB, tA)
                np.clip(iB, 1, len(tB) - 1, out=iB)
                dt_left = np.abs(tB[iB] - tA)
                dt_right = np.abs(tB[iB - 1] - tA)
                return (np.minimum(dt_left, dt_right) <= delt).sum()

            if len(tA) == 0 or len(tB) == 0:
                return np.nan  # skip pairs with no spikes

            PA = spikes_in_delt(tA, tB, delt) / len(tA)
            PB = spikes_in_delt(tB, tA, delt) / len(tB)

            aa = (PA - TB) / (1 - PA * TB) if PA * TB != 1 else 0
            bb = (PB - TA) / (1 - PB * TA) if PB * TA != 1 else 0
            return (aa + bb) / 2

        N = len(spike_train)
        T = length
        ts = [time_in_delt(ts, delt, T) / T for ts in spike_train]

        matrix = np.diag(np.ones(N))
        for i in range(N):
            for j in range(i + 1, N):
                sttc_value = sttc_pairs(spike_train[i], spike_train[j], ts[i], ts[j], delt)
                if not np.isnan(sttc_value):  # Skip NaN values
                    matrix[i, j] = matrix[j, i] = sttc_value
        return matrix

    @staticmethod
    def get_upper_triangle_values(matrix):

        if matrix.size == 0:
            print("Warning: Empty STTC matrix. Returning empty array.")
            return np.array([])

        upper_triangle_indices = np.triu_indices_from(matrix, k=1)
        return matrix[upper_triangle_indices]

    def compute_spike_triggered_sttc(self, train, duration, window_size=1.0):
        spike_windows = []
        for spike_times in train:
            for spike in spike_times:
                start_time = max(0, spike - window_size / 2)
                end_time = min(duration, spike + window_size / 2)
                windowed_train = [neuron_spikes[(neuron_spikes >= start_time) & (neuron_spikes <= end_time)] for neuron_spikes in train]
                sttc_matrix = self.compute_sttc_matrix(windowed_train, duration)
                spike_windows.append(sttc_matrix)
        return spike_windows

    @staticmethod
    def group_neurons_by_firing_rate(firing_rates):
        low_threshold = np.percentile(firing_rates, 33)
        high_threshold = np.percentile(firing_rates, 66)

        groups = {
            'low': [i for i, rate in enumerate(firing_rates) if rate < low_threshold],
            'medium': [i for i, rate in enumerate(firing_rates) if low_threshold <= rate < high_threshold],
            'high': [i for i, rate in enumerate(firing_rates) if rate >= high_threshold]
        }
        return groups

    @staticmethod
    def group_neurons_by_proximity(neuron_data, distance_threshold=100):
        neuron_positions = np.array([neuron_data[i]['position'] for i in range(len(neuron_data))])
        num_neurons = len(neuron_positions)
        distances = np.linalg.norm(neuron_positions[:, np.newaxis] - neuron_positions, axis=2)

        close_indices = set()
        distant_indices = set(range(num_neurons))
        for i in range(num_neurons):
            for j in range(i + 1, num_neurons):
                if distances[i, j] < distance_threshold:
                    close_indices.add(i)
                    close_indices.add(j)
        distant_indices.difference_update(close_indices)

        return {'close': list(close_indices), 'distant': list(distant_indices)}

    def sttc_plot(self, output_path, dataset_name):
        plt.close('all')
        for i, train in enumerate(self.trains):
            matrix = self.compute_sttc_matrix(train, self.durations[i])

            plt.figure(figsize=(10, 8))
            plt.imshow(matrix, cmap='YlOrRd')

            plt.colorbar(label='STTC')
            plt.xlabel('Neuron')
            plt.ylabel('Neuron')

            num_neurons = len(train)
            tick_positions = np.linspace(0, num_neurons - 1, 20, dtype=int)
            plt.xticks(ticks=tick_positions, labels=tick_positions)
            plt.yticks(ticks=tick_positions, labels=tick_positions)

            plt.title(f'Heatmap of Functional Connectivity: {dataset_name}')
            plt.savefig(os.path.join(output_path, f"sttc_heatmap_{dataset_name}.png"))
            plt.close()


    def sttc_violin_plot_by_proximity(self, output_path, dataset_names, distance_threshold=100):
        """
        Generate STTC violin plots by proximity.

        Parameters:
            output_path (str): Directory to save plots.
            dataset_names (list): Names of datasets.
            distance_threshold (int): Threshold for proximity grouping.
        """
        plt.close('all')

        # If the output path is a dataset-specific directory
        if output_path not in [os.path.join(output_path, "comparisons")]:
            for i, train in enumerate(self.trains):
                dataset_name = dataset_names[i]
                groups = self.group_neurons_by_proximity(self.neuron_data_list[i], distance_threshold)
                sttc_values = []
                labels = []

                for group_name, indices in groups.items():
                    if len(indices) < 2:
                        continue
                    sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                    sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                    labels.append(f"{group_name.capitalize()}")

                plt.figure(figsize=(12, 8))
                plt.violinplot(sttc_values, showmeans=True)
                plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha='right')
                plt.xlabel('Proximity Group')
                plt.ylabel('STTC Values')
                plt.title(f'Violin Plot of STTC Values by Spatial Proximity: {dataset_name} (Threshold = {distance_threshold}μm)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_proximity_{dataset_name}.png"))
                plt.close()

        # If the output path is the comparison directory
        if output_path == os.path.join(output_path, "comparisons"):
            combined_sttc_values = []
            combined_labels = []

            for i, train in enumerate(self.trains):
                dataset_name = dataset_names[i]
                groups = self.group_neurons_by_proximity(self.neuron_data_list[i], distance_threshold)

                for group_name, indices in groups.items():
                    if len(indices) < 2:
                        continue
                    sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                    combined_sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                    combined_labels.append(f"{dataset_name} - {group_name.capitalize()}")

            plt.figure(figsize=(12, 8))
            plt.violinplot(combined_sttc_values, showmeans=True)
            plt.xticks(ticks=np.arange(1, len(combined_labels) + 1), labels=combined_labels, rotation=45, ha='right')
            plt.xlabel('Proximity Group')
            plt.ylabel('STTC Values')
            plt.title(f'Violin Plot of STTC Values by Spatial Proximity Across Recordings (Threshold = {distance_threshold}μm)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_proximity_combined.png"))
            plt.close()


    def sttc_violin_plot_by_firing_rate(self, output_path, dataset_names):
        """
        Generate STTC violin plots by firing rate.

        Parameters:
            output_path (str): Directory to save plots.
            dataset_names (list): Names of datasets.
        """
        plt.close('all')

        # If the output path is a dataset-specific directory
        if output_path not in [os.path.join(output_path, "comparisons")]:
            for i, train in enumerate(self.trains):
                dataset_name = dataset_names[i]
                groups = self.group_neurons_by_firing_rate(self.firing_rates_list[i])
                sttc_values = []
                labels = []

                for group_name, indices in groups.items():
                    sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                    sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                    labels.append(f"{group_name.capitalize()}")

                plt.figure(figsize=(12, 8))
                plt.violinplot(sttc_values, showmeans=True)
                plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha='right')
                plt.xlabel('Firing Rate Group')
                plt.ylabel('STTC Values')
                plt.title(f'Violin Plot of STTC Values by Firing Rate: {dataset_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_firing_rate_{dataset_name}.png"))
                plt.close()

        # If the output path is the comparison directory
        if output_path == os.path.join(output_path, "comparisons"):
            combined_sttc_values = []
            combined_labels = []

            for i, train in enumerate(self.trains):
                dataset_name = dataset_names[i]
                groups = self.group_neurons_by_firing_rate(self.firing_rates_list[i])

                for group_name, indices in groups.items():
                    sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                    combined_sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                    combined_labels.append(f"{dataset_name} - {group_name.capitalize()}")

            plt.figure(figsize=(12, 8))
            plt.violinplot(combined_sttc_values, showmeans=True)
            plt.xticks(ticks=np.arange(1, len(combined_labels) + 1), labels=combined_labels, rotation=45, ha='right')
            plt.xlabel('Firing Rate Group')
            plt.ylabel('STTC Values')
            plt.title('Violin Plot of STTC Values by Firing Rate Group Across Recordings')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_firing_rate_combined.png"))
            plt.close()




    def sttc_violin_plot_across_recordings(self, output_path, dataset_names):
        plt.close('all')
        if len(self.input_paths) < 2:
            print("Only one dataset provided. Skipping multi-recording comparison.")
            return

        sttc_values_list = []
        labels = []

        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            sttc_matrix = self.compute_sttc_matrix(train, duration)
            sttc_values = self.get_upper_triangle_values(sttc_matrix)
            sttc_values_list.append(sttc_values)
            labels.append(dataset_names[i])

        plt.figure(figsize=(12, 8))
        plt.violinplot(sttc_values_list, showmeans=True)
        plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha ='right')
        plt.xlabel('Recordings')
        plt.ylabel('STTC Values')
        plt.title('Violin Plot of STTC Values Across Recordings')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_across_recordings.png"))
        plt.close()

    def plot_firing_rate_histogram(self, output_path, dataset_name):
        plt.close('all')
        for i, firing_rates in enumerate(self.firing_rates_list):
            plt.figure(figsize=(12, 6))
            plt.hist(firing_rates, bins=50, color='green', alpha=0.7)
            plt.xlabel('Firing Rate (Hz)')
            plt.ylabel('Count')
            plt.title(f'Firing Rate Histogram: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"firing_rate_histogram_{dataset_name}.png"))
            plt.close()

    def plot_firing_rate_cdf(self, output_path, dataset_name):
        plt.close('all')
        for i, firing_rates in enumerate(self.firing_rates_list):
            sorted_firing_rates = np.sort(firing_rates)
            cdf = np.arange(len(sorted_firing_rates)) / float(len(sorted_firing_rates))
        
            plt.figure(figsize=(12, 6))
            plt.plot(sorted_firing_rates, cdf, color='purple')
            plt.xlabel('Firing Rate (Hz)')
            plt.ylabel('CDF')
            plt.title(f'Cumulative Distribution Function of Firing Rates: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"firing_rate_cdf_{dataset_name}.png"))
            plt.close()

    def plot_isi_histogram(self, output_path, dataset_name):
        plt.close('all')
        for i, train in enumerate(self.trains):
            all_intervals = []
            for neuron_spikes in train:
                all_intervals.extend(np.diff(neuron_spikes))

            plt.figure(figsize=(12, 6))
            plt.hist(all_intervals, bins=50, color='cyan', alpha=0.7)
            plt.xlabel('Inter-Spike Interval (s)')
            plt.ylabel('Count')
            plt.title(f'ISI Histogram: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"isi_histogram_{dataset_name}.png"))
            plt.close()

    def plot_cv_of_isi(self, output_path, dataset_name):
        plt.close('all')
        for i, train in enumerate(self.trains):
            cv_values = []
            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)
                if len(intervals) > 1:
                    cv_values.append(np.std(intervals) / np.mean(intervals))  # CV: std/mean

            plt.figure(figsize=(12, 6))
            plt.hist(cv_values, bins=50, color='orange', alpha=0.7)
            plt.xlabel('Coefficient of Variation (CV)')
            plt.ylabel('Count')
            plt.title(f'Coefficient of Variation of ISI: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"cv_of_isi_{dataset_name}.png"))
            plt.close()

    def plot_raw_population_fr(self, output_path, dataset_name, bin_size=1.0):
        plt.close('all')
        for i, train in enumerate(self.trains):
            combined_train = np.hstack(train)
            max_time = np.max(combined_train)
            bins = np.arange(0, max_time + bin_size, bin_size)
            firing_rate, _ = np.histogram(combined_train, bins=bins)

            plt.figure(figsize=(12, 6))
            plt.plot(bins[:-1], firing_rate, color='red')
            plt.xlabel('Time (s)')
            plt.ylabel('Raw Population Firing Rate (Hz)')
            plt.title(f'Raw Population Firing Rate: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"raw_population_fr_{dataset_name}.png"))
            plt.close()

    def plot_synchrony_index_over_time(self, output_path, dataset_name, window_size=10.0):
        plt.close('all')
        for i, train in enumerate(self.trains):
            synchrony_indices = []
            time_points = []
            max_time = np.max(np.hstack(train))
            current_time = 0

            while current_time + window_size <= max_time:
                windowed_trains = [
                    spike_times[(spike_times >= current_time) & (spike_times < current_time + window_size)] 
                    for spike_times in train
                ]
            
                # pad arrays to the same length
                max_len = max(len(trains) for trains in windowed_trains)
                padded_trains = np.array([np.pad(trains, (0, max_len - len(trains)), 'constant') for trains in windowed_trains])

                # skip if all padded arrays are zeros or contain only a single unique value
                if np.all(padded_trains == 0) or np.all(padded_trains == padded_trains[0]):
                    current_time += window_size
                    continue
            
                # compute correlation if there are more than 1 valid time series
                if padded_trains.shape[0] > 1:
                    pairwise_corr = np.corrcoef(padded_trains)
                    if not np.isnan(pairwise_corr).any():  # skip if correlation contains NaNs
                        synchrony_indices.append(np.mean(pairwise_corr[np.triu_indices_from(pairwise_corr, k=1)]))
                        time_points.append(current_time + window_size / 2)

                current_time += window_size

            if synchrony_indices:
                plt.figure(figsize=(12, 6))
                plt.plot(time_points, synchrony_indices, color='magenta')
                plt.xlabel('Time (s)')
                plt.ylabel('Synchrony Index')
                plt.title(f'Synchrony Index Over Time: {dataset_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"synchrony_index_over_time_{dataset_name}.png"))
                plt.close()
            else:
                print(f"No valid synchrony data for plotting for dataset: {dataset_name}")

    def plot_active_units_per_electrode(self, output_path, dataset_name):
        plt.close('all')
        for i, neuron_data in enumerate(self.neuron_data_list):
            electrode_activity = np.zeros(len(neuron_data))

            for j, neuron in enumerate(neuron_data.values()):
                # check if 'spike_times' key exists and is not empty
                if 'spike_times' in neuron and neuron['spike_times'] is not None and len(neuron['spike_times']) > 0:
                    electrode_activity[j] += 1

            plt.figure(figsize=(12, 6))
            plt.bar(np.arange(len(electrode_activity)), electrode_activity, color='teal')
            plt.xlabel('Electrode Index')
            plt.ylabel('Number of Active Units')
            plt.title(f'Active Units Per Electrode: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"active_units_per_electrode_{dataset_name}.png"))
            plt.close()

    def plot_electrode_activity_heatmap(self, output_path, dataset_name):
        plt.close('all')
        for i, neuron_data in enumerate(self.neuron_data_list):
            electrode_activity = np.zeros(len(neuron_data))

            for j, neuron in enumerate(neuron_data.values()):
                # check if 'spike_times' key exists and is not empty
                if 'spike_times' in neuron and neuron['spike_times'] is not None:
                    electrode_activity[j] += len(neuron['spike_times'])

            # attempt to create a heatmap without reshaping
            plt.figure(figsize=(12, 6))
            plt.imshow([electrode_activity], aspect='auto', cmap='hot', interpolation='nearest')
            plt.colorbar(label='Activity Level')
            plt.xlabel('Electrode Index')
            plt.title(f'Electrode Activity Heatmap: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"electrode_activity_heatmap_{dataset_name}.png"))
            plt.close()



    def plot_sttc_over_time(self, output_path, dataset_name, window_size=10.0):
        plt.close('all')
        for i, train in enumerate(self.trains):
            sttc_values = []
            time_points = []
            max_time = np.max(np.hstack(train))
            current_time = 0

            while current_time + window_size <= max_time:
                windowed_trains = [spike_times[(spike_times >= current_time) & (spike_times < current_time + window_size)] for spike_times in train]
                if len(windowed_trains) > 1:
                    sttc_matrix = self.compute_sttc_matrix(windowed_trains, window_size)
                    sttc_values.append(np.mean(self.get_upper_triangle_values(sttc_matrix)))
                    time_points.append(current_time + window_size / 2)

                current_time += window_size

            plt.figure(figsize=(12, 6))
            plt.plot(time_points, sttc_values, color='blue')
            plt.xlabel('Time (s)')
            plt.ylabel('STTC')
            plt.title(f'STTC Over Time: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"sttc_over_time_{dataset_name}.png"))
            plt.close()
    
    def plot_footprint_sttc(self, output_path, dataset_name):
        plt.close('all')
        for i, (train, neuron_data) in enumerate(zip(self.trains, self.neuron_data_list)):

            neuron_x, neuron_y, sttc_marker_size = [], [], []

            sttc_matrix = self.compute_sttc_matrix(train, self.durations[i])

            sttc_sums = np.sum(sttc_matrix, axis=1) - np.diag(sttc_matrix) #calculate sums minus the self connection

            for j, neuron in enumerate(neuron_data.values()):
                x, y = neuron['position']
                if (x, y) != (0, 0):  # exclude the calibration electrode
                    neuron_x.append(x)
                    neuron_y.append(y)
                    sttc_marker_size.append(sttc_sums[j])

            legend_rates = np.percentile(sttc_sums, [50, 75, 90, 98])

            plt.figure(figsize=(11, 9))
            plt.scatter(neuron_x, neuron_y, s=sttc_marker_size * 100, alpha=0.4, c='b', edgecolors='none')
        
            for rate in legend_rates:
                plt.scatter([], [], s=rate * 100, c='r', alpha=0.4, label=f'STTC Sum {rate /100:.2f}')

            plt.legend(scatterpoints=1, frameon=True, labelspacing=1.4, handletextpad=0.8, borderpad=0.92, title='STTC Sum', loc = 'best', title_fontsize=10, fontsize=10)
            plt.xlabel(r"Horizontal Position ($\mu$m)")
            plt.ylabel(r"Vertical Position ($\mu$m)")
            plt.title(f"Footprint Plot with STTC Sum : {dataset_name}")
            plt.savefig(os.path.join(output_path, f"footprint_plot_sttc_sum_{dataset_name}.png"))
            plt.close()

    def plot_comparison_inverse_isi(self, output_path, base_names):
        """
        Generate and save a comparison overlay plot for population-level inverse ISI for all datasets.
        """
        plt.figure(figsize=(12, 6))
        for i, train in enumerate(self.trains):
            all_intervals = []
            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)
                if len(intervals) > 0:
                    all_intervals.extend(intervals)
            inverse_isi = 1 / np.array(all_intervals)
            plt.hist(inverse_isi, bins=100, alpha=0.5, label=base_names[i], density=True)

        plt.xlabel("Instantaneous Firing Rate (Hz)")
        plt.ylabel("Density")
        plt.title("Comparison: Population-Level Inverse ISI Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "comparison_population_inverse_isi.png"))
        plt.close()

    def plot_comparison_regular_isi(self, output_path, base_names):
        """
        Generate and save a comparison overlay plot for regular ISI histograms for all datasets.
        """
        plt.figure(figsize=(12, 6))
        for i, train in enumerate(self.trains):
            all_intervals = []
            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)
                if len(intervals) > 0:
                    all_intervals.extend(intervals)
            plt.hist(all_intervals, bins=100, alpha=0.5, label=base_names[i], density=True)

        plt.xlabel("Inter-Spike Interval (s)")
        plt.ylabel("Density")
        plt.title("Comparison: Regular ISI Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "comparison_regular_isi.png"))
        plt.close()

    def find_neuron_of_interest(self, dataset_index=0):
        """
        Identify the neuron of interest based on composite ranking criteria.
        Criteria: 
        1. Firing rate (higher is better).
        2. Regular spiking intervals (lower CV is better).
        3. Tendency to fire in bursts (higher is better).
        4. Total STTC (higher is better).
        
        Returns:
            The index of the "neuron of interest."
        """
        if dataset_index >= len(self.trains):
            print(f"Dataset index {dataset_index} out of bounds.")
            return None

        train = self.trains[dataset_index]
        firing_rates = self.firing_rates_list[dataset_index]
        duration = self.durations[dataset_index]

        if len(train) == 0 or len(firing_rates) == 0:
            print("No neurons in dataset.")
            return None

        # initialize rank arrays
        num_neurons = len(train)
        firing_rate_ranks = np.argsort(-np.array(firing_rates))  # descending order
        isi_cvs = []
        burst_scores = []
        sttc_ranks = np.zeros(num_neurons)

        # compute ISI CV
        for neuron_spikes in train:
            intervals = np.diff(neuron_spikes)
            if len(intervals) > 1:
                isi_cvs.append(np.std(intervals) / np.mean(intervals))  # CV = std/mean
            else:
                isi_cvs.append(np.inf)
        isi_cv_ranks = np.argsort(isi_cvs)

        # compute burst tendency
        for neuron_spikes in train:
            intervals = np.diff(neuron_spikes)
            burst_scores.append(np.sum(intervals < 0.1))  # count intervals < 100ms as bursts
        burst_ranks = np.argsort(-np.array(burst_scores))

        # compute STTC
        sttc_matrix = self.compute_sttc_matrix(train, duration)
        total_sttc_scores = np.sum(sttc_matrix, axis=0) - 1  # exclude self-connections
        sttc_ranks = np.argsort(-np.array(total_sttc_scores))

        # combine ranks
        composite_ranks = firing_rate_ranks + isi_cv_ranks + burst_ranks + sttc_ranks
        neuron_of_interest = np.argmin(composite_ranks)
        print(f"Neuron of interest for dataset {dataset_index}: {neuron_of_interest}")
        return neuron_of_interest

    def plot_population_inverse_isi(self, output_path, dataset_name):
        for i, train in enumerate(self.trains):
            all_intervals = []
            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)
                if len(intervals) > 0:
                    all_intervals.extend(intervals)
            inverse_isi = 1 / np.array(all_intervals)

            plt.figure(figsize=(12, 6))
            plt.hist(inverse_isi, bins=100, color='green', alpha=0.7, density=True)
            plt.xlabel("Instantaneous Firing Rate (Hz)")
            plt.ylabel("Density")
            plt.title(f"Population-Level Inverse ISI: {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"population_inverse_isi_{dataset_name}.png"))
            plt.close()

    def plot_neuron_inverse_isi(self, output_path, dataset_name, dataset_index=0):
        neuron_index = self.find_neuron_of_interest(dataset_index)
        if neuron_index is None:
            print(f"Could not determine neuron of interest for dataset {dataset_name}. Skipping.")
            return
        train = self.trains[dataset_index]
        intervals = np.diff(train[neuron_index])
        if len(intervals) > 0:
            inverse_isi = 1 / intervals

            plt.figure(figsize=(12, 6))
            plt.hist(inverse_isi, bins=100, color='blue', alpha=0.7, density=True)
            plt.xlabel("Instantaneous Firing Rate (Hz)")
            plt.ylabel("Density")
            plt.title(f"Neuron {neuron_index} Inverse ISI: {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"neuron_{neuron_index}_inverse_isi_{dataset_name}.png"))
            plt.close()

    def plot_neuron_regular_isi(self, output_path, dataset_name, dataset_index=0):
        neuron_index = self.find_neuron_of_interest(dataset_index)
        if neuron_index is None:
            print(f"Could not determine neuron of interest for dataset {dataset_name}. Skipping.")
            return
        train = self.trains[dataset_index]
        intervals = np.diff(train[neuron_index])
        if len(intervals) > 0:
            plt.figure(figsize=(12, 6))
            plt.hist(intervals, bins=100, color='orange', alpha=0.7, density=True)
            plt.xlabel("Inter-Spike Interval (s)")
            plt.ylabel("Density")
            plt.title(f"Neuron {neuron_index} Regular ISI: {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"neuron_{neuron_index}_regular_isi_{dataset_name}.png"))
            plt.close()

    @staticmethod
    def calculate_pca(firing_rate_matrix, subset_neurons=None):
        from sklearn.decomposition import PCA
        """
        Calculate PCA for a given firing rate matrix.

        Parameters:
            firing_rate_matrix (numpy.ndarray): Rows = Neurons, Columns = Conditions (datasets).
            subset_neurons (int, optional): Number of top neurons (by mean firing rate) to include.
                                            If None, all neurons are included.

        Returns:
            tuple: (pca_result, explained_variance, pca_object)
                - pca_result: Transformed PCA coordinates (Neurons x PCs).
                - explained_variance: Variance explained by each PC.
                - pca_object: Fitted PCA object.
        """
        if np.all(firing_rate_matrix == firing_rate_matrix[0, :]):
            print("Firing rate matrix has no variance. PCA computation skipped.")
            return np.zeros_like(firing_rate_matrix), np.zeros(firing_rate_matrix.shape[1]), None

        # optionally filter top neurons by mean firing rate
        if subset_neurons:
            mean_firing_rates = np.mean(firing_rate_matrix, axis=1)
            top_indices = np.argsort(-mean_firing_rates)[:subset_neurons]  # Descending order
            firing_rate_matrix = firing_rate_matrix[top_indices, :]

        # perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(firing_rate_matrix)
        explained_variance = pca.explained_variance_ratio_

        return pca_result, explained_variance, pca
    
    def prepare_pca(self, firing_rates_list, subset_neurons=None):
        """
        Prepare the firing rate matrix and calculate PCA.

        Parameters:
            firing_rates_list (list of list): Firing rates for all datasets.
            subset_neurons (int, optional): Number of top neurons (by mean firing rate) to include.
                                            If None, all neurons are included.

        Returns:
            tuple: (pca_result, explained_variance)
                - pca_result: Transformed PCA coordinates (Neurons x PCs).
                - explained_variance: Variance explained by each PC.
        """
        # prepare firing rate matrix: rows = neurons, columns = conditions
        all_firing_rates = [np.array(firing_rates) for firing_rates in firing_rates_list]
        firing_rate_matrix = np.vstack(all_firing_rates).T

        # calculate PCA
        pca_result, explained_variance, _ = SpikeDataAnalysis.calculate_pca(firing_rate_matrix, subset_neurons=subset_neurons)

        return pca_result, explained_variance

    # define PCA plotting function
    def plot_pca(self, pca_result, explained_variance, output_path, base_names):
        """
        Plot PCA results: scatter plot (PC1 vs PC2) and scree plot.

        Parameters:
            pca_result (numpy.ndarray): Transformed PCA coordinates (Neurons x PCs).
            explained_variance (numpy.ndarray): Variance explained by each PC.
            output_path (str): Directory to save the PCA plots.
            base_names (list of str): Names of the datasets/conditions.
        """
        # scatter Plot: neurons in PCA space (PC1 vs PC2)
        plt.figure(figsize=(10, 8))
        for i in range(len(base_names)):
            plt.scatter(
                pca_result[:, 0][i::len(base_names)],
                pca_result[:, 1][i::len(base_names)],
                label=base_names[i],
                alpha=0.7
            )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA of Firing Rate Variance (PC1 vs PC2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "pca_scatter_plot.png"))
        plt.close()

        # scree plot: variance explained by each PC
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.xlabel("Principal Component")
        plt.ylabel("Proportion of Variance Explained")
        plt.title("Scree Plot of PCA Components")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "pca_scree_plot.png"))
        plt.close()

    def plot_high_activity_rasters(self, output_path, dataset_name):
        """
        Generate raster plots for high-activity periods with smaller time windows.

        Parameters:
            output_path (str): Directory to save the raster plots.
            dataset_name (str): Name of the dataset.

        Saves:
            Raster plots for specified time windows centered on high-activity periods.
        """
        for i, train in enumerate(self.trains):
            # calculate population firing rate
            bins, fr_avg = self.get_population_fr(train, average=True)

            #error handling
            if len(fr_avg) == 0 or np.all(fr_avg == 0):
                print(f"No activity detected in dataset {dataset_name}. Skipping high-activity rasters.")
                continue
            
            # identify high-activity periods
            threshold = np.percentile(fr_avg, 95) #changed to 95 from 90 to decrease plots created
            high_activity_times = bins[np.where(fr_avg > threshold)]

            #error handling
            if len(high_activity_times) == 0:
                print(f"No high-activity periods found for dataset {dataset_name}.")
                continue

            # create output subdirectory for short-window rasters
            raster_output_dir = os.path.join(output_path, f"rasters/short_windows")
            os.makedirs(raster_output_dir, exist_ok=True)

            # define the smaller raster window sizes
            window_sizes = [60, 30, 10]

            for window_size in window_sizes:
                for center_time in high_activity_times:
                    start_time = max(0, center_time - window_size / 2)
                    end_time = start_time + window_size

                    # generate raster plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    y = 0
                    for vv in train:
                        spikes_in_window = vv[(vv >= start_time) & (vv <= end_time)]
                        ax.scatter(spikes_in_window, [y] * len(spikes_in_window), marker="|", c='k', s=4, alpha=0.7)
                        y += 1

                    # overlay population firing rate
                    bins_in_window = (bins >= start_time) & (bins <= end_time)
                    ax2 = ax.twinx()
                    ax2.plot(bins[bins_in_window], fr_avg[bins_in_window], color='red', alpha=0.5, linewidth=2)
                    ax2.set_ylabel("Population Firing Rate (Hz)", color='red')

                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Neuron Index")
                    ax.set_xlim(start_time, end_time)
                    ax.set_title(f"Raster Plot ({window_size}s) - High Activity: {dataset_name}")

                    #redefined a plot_filename for debugging. try to use this in future
                    raster_output_dir = os.path.join(output_path, f"rasters/high_activity_raster/{window_size}s")
                    os.makedirs(raster_output_dir, exist_ok=True)
                    plot_filename = os.path.join(raster_output_dir, f"raster_{dataset_name}_window_{window_size}s_{int(center_time)}s.png")
                    plt.tight_layout()
                    plt.savefig(plot_filename)
                    plt.close(fig)


    def plot_comparison_firing_rate_histogram(self, output_path, base_names, bins=50):
        """
        Generate and save a comparison overlay plot for firing rate histograms for all datasets.

        Parameters:
            output_path (str): Directory to save the comparison plot.
            base_names (list of str): Names of the datasets for labeling.

        Saves:
            A combined histogram of firing rates overlayed for all datasets.
        """

        #calculate the min/mac firing rate over all dataset
        all_firing_rates = np.hstack(self.firing_rates_list)
        min_firing_rate = np.min(all_firing_rates)
        max_firing_rate = np.max(all_firing_rates)

        bin_edges = np.linspace(min_firing_rate, max_firing_rate, bins+1)  # 50 bins

        plt.figure(figsize=(12, 6))
        for i, firing_rates in enumerate(self.firing_rates_list):
            plt.hist(
                firing_rates,
                bins=bin_edges,
                alpha=0.5,
                label=base_names[i],
                density=True
            )

        plt.xlabel("Firing Rate (Hz)")
        plt.ylabel("Density")
        plt.title("Comparison: Firing Rate Histograms Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "comparison_firing_rate_histogram.png"))
        plt.close()

    def plot_sttc_log(self, output_path, dataset_name):
        from matplotlib.colors import LogNorm
        """
        Generate an STTC heatmap on a logarithmic scale for all datasets.
        """
        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            sttc_matrix = self.compute_sttc_matrix(train, duration)

            plt.figure(figsize=(10, 8))
            log_norm = LogNorm(vmin=max(np.min(sttc_matrix), 1e-6), vmax=np.max(sttc_matrix))
            plt.imshow(sttc_matrix, cmap='coolwarm', norm=log_norm)
            plt.colorbar(label='Log-STTC')
            plt.xlabel('Neuron')
            plt.ylabel('Neuron')
            plt.title(f'STTC Heatmap (Log Scale): {dataset_name}')
            plt.tight_layout()

            plot_name = os.path.join(output_path, f"sttc_log_{dataset_name}.png") #try to do this in the future, maybe easier to read
            plt.savefig(plot_name)
            plt.close()

    def plot_sttc_vmin_vmax(self, output_path, dataset_name):
        """
        Generate an STTC heatmap with dynamic vmin and vmax boundaries for all datasets.
        """
        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            sttc_matrix = self.compute_sttc_matrix(train, duration)
            vmin = np.percentile(sttc_matrix, 5)
            vmax = np.percentile(sttc_matrix, 95)

            plt.figure(figsize=(10, 8))
            plt.imshow(sttc_matrix, cmap='coolwarm', vmin=vmin, vmax=vmax)
            plt.colorbar(label='STTC')
            plt.xlabel('Neuron')
            plt.ylabel('Neuron')
            plt.title(f'STTC Heatmap (Dynamic Range): {dataset_name}')
            plt.tight_layout()

            plot_name = os.path.join(output_path, f"sttc_vmin_vmax_{dataset_name}.png")
            plt.savefig(plot_name)
            plt.close()

    def plot_sttc_thresh(self, output_path, dataset_name):
        from matplotlib import cm
        """
        Generate an STTC heatmap with threshold-based shading for all datasets.
        """
        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            sttc_matrix = self.compute_sttc_matrix(train, duration)
            threshold_lower = np.percentile(np.abs(sttc_matrix), 25)

            masked_matrix = np.ma.masked_where(np.abs(sttc_matrix) < threshold_lower, sttc_matrix)

            plt.figure(figsize=(10, 8))
            cmap = cm.get_cmap('coolwarm')  # get the colormap
            cmap.set_bad(color='lightgrey')  # set masked values to grey
            plt.imshow(masked_matrix, cmap=cmap)
            plt.colorbar(label='STTC')
            plt.xlabel('Neuron')
            plt.ylabel('Neuron')
            plt.title(f'STTC Heatmap (Thresholded): {dataset_name}')
            plt.tight_layout()

            plot_name = os.path.join(output_path, f"sttc_thresh_{dataset_name}.png")
            plt.savefig(plot_name)
            plt.close()

    def plot_kde_pdf(self, output_path, dataset_name):
        """
        Generate KDE and PDF plots for STTC values for all datasets.
        """
        for train, duration, name in zip(self.trains, self.durations, dataset_name):
            sttc_matrix = self.compute_sttc_matrix(train, duration)
            sttc_values = self.get_upper_triangle_values(sttc_matrix)
            if sttc_values.size == 0:
                print(f"Warning: No valid STTC values for dataset {dataset_name}. Skipping plot.")
                continue

            # KDE computation
            kde = scipy.stats.gaussian_kde(sttc_values)
            x_range = np.linspace(min(sttc_values), max(sttc_values), 500)
            kde_values = kde(x_range)

            # plot KDE and PDF
            plt.figure(figsize=(10, 6))
            plt.plot(x_range, kde_values, label='KDE', linewidth=2)
            plt.hist(sttc_values, bins=50, density=True, alpha=0.5, label='PDF')

            # highlight specific ranges
            plt.axvline(x=0.2, color='orange', linestyle='--', label='Threshold: |STTC|=0.2')
            plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold: |STTC|=0.5')

            plt.xlabel('STTC Values')
            plt.ylabel('Density')
            plt.title(f'STTC KDE and PDF: {name}')
            plt.legend()
            plt.tight_layout()

            plot_name = os.path.join(output_path, f"kde_pdf_{name}.png")
            plt.savefig(plot_name)
            plt.close()
    def plot_comparison_kde_pdf(self, output_path, base_names):
        """
        Generate comparison KDE/PDF overlay plots for all datasets.
        """
        plt.figure(figsize=(12, 6))

        for train, duration, name in zip(self.trains, self.durations, base_names):
            sttc_matrix = self.compute_sttc_matrix(train, duration)
            sttc_values = self.get_upper_triangle_values(sttc_matrix)

            if sttc_values.size == 0:
                print(f"Warning: No valid STTC values for dataset {dataset_name}. Skipping plot.")
                continue

            # KDE computation
            kde = scipy.stats.gaussian_kde(sttc_values)
            x_range = np.linspace(min(sttc_values), max(sttc_values), 500)
            kde_values = kde(x_range)

            # overlay KDE and PDF
            plt.plot(x_range, kde_values, label=f'{name} (KDE)', alpha=0.7)
            plt.hist(sttc_values, bins=50, density=True, alpha=0.3, label=f'{name} (PDF)')

        # add threshold annotations
        plt.axvline(x=0.2, color='orange', linestyle='--', label='Threshold: |STTC|=0.2')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold: |STTC|=0.5')

        plt.xlabel('STTC Values')
        plt.ylabel('Density')
        plt.title('Comparison: STTC KDE and PDF Across Datasets')
        plt.legend()
        plt.tight_layout()

        plot_name = os.path.join(output_path, "comparison_kde_pdf.png")
        plt.savefig(plot_name)
        plt.close()


    def plot_pairwise_linear_comparison(self, output_path, base_names, metric):
        """
        Pairwise linear comparison of a given metric (e.g., STTC, ISI, Firing Rate).

        Args:
            output_path (str): Path to save the plots.
            base_names (list): Names of the datasets.
            metric (str): Metric to compute and compare ("sttc", "isi", "firing_rate").
        """

        from itertools import combinations

        # metric computation logic
        def compute_metric(metric):
            if metric == "sttc":
                values = []
                for train, duration in zip(self.trains, self.durations):
                    sttc_matrix = self.compute_sttc_matrix(train, duration)
                    values.append(self.get_upper_triangle_values(sttc_matrix))
                return values
            elif metric == "isi":
                values = []
                for train in self.trains:
                    all_intervals = []
                    for neuron_spikes in train:
                        all_intervals.extend(np.diff(neuron_spikes))
                    values.append(np.array(all_intervals))
                return values
            elif metric == "firing_rate":
                return self.firing_rates_list
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        # compute metric values for all datasets
        metric_values_list = compute_metric(metric)

        # pairwise comparisons
        for (i, j) in combinations(range(len(base_names)), 2):
            dataset_1 = metric_values_list[i]
            dataset_2 = metric_values_list[j]
            name_1 = base_names[i]
            name_2 = base_names[j]

            # Scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(dataset_1, dataset_2, alpha=0.7, label=f"{name_1} vs {name_2}")
            plt.plot([0, max(dataset_1 + dataset_2)], [0, max(dataset_1 + dataset_2)], 'r--', label="Equality Line")
            plt.xlabel(f"{name_1} {metric.capitalize()}")
            plt.ylabel(f"{name_2} {metric.capitalize()}")
            plt.title(f"Pairwise Linear Comparison: {metric.capitalize()} ({name_1} vs {name_2})")
            plt.legend()
            plt.tight_layout()
            plot_filename = os.path.join(output_path, f"{metric}_pairwise_{name_1}_vs_{name_2}.png")
            plt.savefig(plot_filename)
            plt.close()

    def plot_global_linear_comparison(self, output_path, base_names, metric):
        """
        Global linear comparison of a given metric across all datasets.

        Args:
            output_path (str): Path to save the plots.
            base_names (list): Names of the datasets.
            metric (str): Metric to compute and compare ("sttc", "isi", "firing_rate").
        """
        # skip if fewer than three datasets 
        if len(base_names) < 3:
            print("Global comparison requires at least three datasets. Skipping.")
            return

        # metric computation logic
        def compute_metric(metric):
            if metric == "sttc":
                values = []
                for train, duration in zip(self.trains, self.durations):
                    sttc_matrix = self.compute_sttc_matrix(train, duration)
                    values.append(self.get_upper_triangle_values(sttc_matrix))
                return values
            elif metric == "isi":
                values = []
                for train in self.trains:
                    all_intervals = []
                    for neuron_spikes in train:
                        all_intervals.extend(np.diff(neuron_spikes))
                    values.append(np.array(all_intervals))
                return values
            elif metric == "firing_rate":
                return self.firing_rates_list
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        # compute metric values for all datasets
        metric_values_list = compute_metric(metric)

        # global comparison: scatter plots for each condition against the other two
        plt.figure(figsize=(12, 8))
        for i, (dataset, name) in enumerate(zip(metric_values_list, base_names)):
            plt.scatter([name] * len(dataset), dataset, label=name, alpha=0.7)

        # add labels and save the plot
        plt.xlabel("Condition")
        plt.ylabel(f"{metric.capitalize()} Values")
        plt.title(f"Global Linear Comparison: {metric.capitalize()} Across All Conditions")
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(output_path, f"{metric}_global_comparison.png")
        plt.savefig(plot_filename)
        plt.close()

    def calculate_continuous_pca(self, bin_size=0.1):
        """
        Calculate PCA on time-binned firing rates for continuous analysis.

        Parameters:
            bin_size (float): Time bin size in seconds.

        Returns:
            tuple: (pca_continuous_result, explained_variance_continuous, time_bins_continuous)
                - pca_continuous_result: Transformed PCA coordinates (time bins x PCs).
                - explained_variance_continuous: Variance explained by each PC.
                - time_bins_continuous: Array of time bin centers.
        """
        from sklearn.decomposition import PCA

        # determine the number of bins and create bin edges
        max_time = max(self.durations)
        num_bins = int(max_time // bin_size) + 1
        bin_edges = np.linspace(0, max_time, num_bins)

        # compute time-binned firing rates for all datasets
        time_binned_firing_rates = []
        for train in self.trains:
            binned_rates = np.array([
                np.histogram(neuron_spikes, bins=bin_edges)[0] / bin_size
                for neuron_spikes in train
            ])
            time_binned_firing_rates.append(binned_rates.T)  # transpose for PCA (rows = time bins)

        # combine datasets into one matrix for PCA
        combined_matrix = np.vstack(time_binned_firing_rates)

        # perform PCA
        pca = PCA()
        pca_continuous_result = pca.fit_transform(combined_matrix)
        explained_variance_continuous = pca.explained_variance_ratio_

        # calculate time bin centers for visualization
        time_bins_continuous = bin_edges[:-1] + (bin_size / 2)

        return pca_continuous_result, explained_variance_continuous, time_bins_continuous

    def plot_continuous_pca(self, pca_continuous_result, explained_variance_continuous, time_bins_continuous, output_path, base_names):
        """
        Plot PCA results for continuous analysis.

        Parameters:
            pca_continuous_result: PCA-transformed data (time bins x PCs).
            explained_variance_continuous: Variance explained by each PC.
            time_bins_continuous: Array of time bin centers.
            output_path: Directory to save the plots (comparison directory).
            base_names: Names of datasets for labeling.
        """
        os.makedirs(output_path, exist_ok=True)

        # scatter plot: PC1 vs PC2 (colored by time)
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_continuous_result[:, 0], pca_continuous_result[:, 1], c=np.tile(time_bins_continuous, len(base_names)), cmap='viridis', alpha=0.7)
        plt.colorbar(label='Time (s)')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Continuous PCA: PC1 vs PC2 (Colored by Time)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "continuous_pca_scatter.png"))
        plt.close()

        # component time series (first few PCs)
        plt.figure(figsize=(12, 6))
        for i in range(3):  # plot first 3 PCs
            plt.plot(time_bins_continuous, pca_continuous_result[:, i], label=f"PC{i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Component Value")
        plt.title("PCA Components Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "continuous_pca_time_series.png"))
        plt.close()

        # scree plot: explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance_continuous) + 1), explained_variance_continuous, alpha=0.7)
        plt.xlabel("Principal Component")
        plt.ylabel("Proportion of Variance Explained")
        plt.title("Scree Plot of Continuous PCA Components")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "continuous_pca_scree_plot.png"))
        plt.close()

    def analyze_burst_characteristics(self, output_path, dataset_names):
        """
        Analyze and compare burst characteristics (inter-burst intervals, burst frequencies, durations, and neuron participation) across datasets.

        Parameters:
            output_path (str): Directory to save the analysis outputs.
            dataset_names (list): Names of the datasets being compared.

        Saves:
            Comparison plots for burst characteristics in the comparison directory.
        """
        # Use the pre-existing comparison directory
        comparison_dir = os.path.join(output_path, "comparisons", "burst_characteristics")
        os.makedirs(comparison_dir, exist_ok=True)

        # Initialize storage for comparisons across datasets
        all_burst_intervals = []
        all_burst_frequencies = []
        all_burst_durations = []
        all_neuron_participation = []

        for i, train in enumerate(self.trains):
            burst_intervals = []
            burst_frequencies = []
            burst_durations = []
            neuron_participation = []

            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)

                # Inter-burst intervals
                if intervals.size > 0:
                    burst_intervals.extend(intervals)

                    # Burst frequency (inverse of the mean interval)
                    burst_frequencies.append(1 / np.mean(intervals))

                    # Burst duration (sum of intervals between spikes in a burst)
                    burst_durations.append(np.sum(intervals))

                # Neuron participation (count neurons with spikes in the burst window)
                burst_spike_counts = np.histogram(neuron_spikes, bins=np.arange(0, max(neuron_spikes), 1))[0]
                neuron_participation.append(np.count_nonzero(burst_spike_counts > 0))

            all_burst_intervals.append(burst_intervals)
            all_burst_frequencies.append(burst_frequencies)
            all_burst_durations.append(burst_durations)
            all_neuron_participation.append(neuron_participation)

        # Comparison Plots
        # Inter-Burst Interval Comparison
        plt.figure(figsize=(12, 6))
        for i, burst_intervals in enumerate(all_burst_intervals):
            plt.hist(burst_intervals, bins=50, alpha=0.5, label=dataset_names[i], density=True)
        plt.xlabel('Inter-Burst Interval (s)')
        plt.ylabel('Density')
        plt.title('Comparison: Inter-Burst Interval Across Datasets')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "comparison_inter_burst_interval.png"))
        plt.close()

        # Burst Frequency vs. Duration Comparison
        plt.figure(figsize=(12, 6))
        for i, (burst_frequencies, burst_durations) in enumerate(zip(all_burst_frequencies, all_burst_durations)):
            plt.scatter(burst_frequencies, burst_durations, alpha=0.6, label=dataset_names[i])
        plt.xlabel('Burst Frequency (Hz)')
        plt.ylabel('Burst Duration (s)')
        plt.title('Comparison: Burst Frequency vs. Duration Across Datasets')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "comparison_burst_freq_duration.png"))
        plt.close()

        # Neuron Participation Comparison
        plt.figure(figsize=(12, 6))
        for i, neuron_participation in enumerate(all_neuron_participation):
            plt.hist(neuron_participation, bins=50, alpha=0.5, label=dataset_names[i], density=True)
        plt.xlabel('Neuron Participation Count')
        plt.ylabel('Density')
        plt.title('Comparison: Neuron Participation Across Datasets')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "comparison_neuron_participation.png"))
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
            rasters_dir = os.path.join(dataset_dir, "rasters")
            os.makedirs(rasters_dir, exist_ok=True)

            dataset_directories.append((dataset_dir, rasters_dir, base_name))

        # create a directory for comparison plots
        comparison_dir = os.path.join(output_folder, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)

        #create subdirectories inside comparison to avoid crowding with plots of similar names 
        pairwise_dir = os.path.join(comparison_dir, "linear_comparisons", "pairwise")
        global_dir = os.path.join(comparison_dir, "linear_comparisons", "global")
        os.makedirs(pairwise_dir, exist_ok=True)
        os.makedirs(global_dir, exist_ok=True)

        # perform analyses for each dataset
        for i, (dataset_dir, rasters_dir, dataset_name) in enumerate(dataset_directories):
            
            #create rasters 
            self.raster_plot(rasters_dir, dataset_name)
            self.plot_high_activity_rasters(rasters_dir, dataset_name)

            #regular analysis
            self.footprint_opaque_circles(dataset_dir, dataset_name)
            self.overlay_fr_raster(dataset_dir, dataset_name)
            self.plot_smoothed_population_fr(dataset_dir, dataset_name)
            self.sttc_plot(dataset_dir, dataset_name)
            self.plot_firing_rate_histogram(dataset_dir, dataset_name)
            self.plot_firing_rate_cdf(dataset_dir, dataset_name)
            self.plot_isi_histogram(dataset_dir, dataset_name)
            self.plot_cv_of_isi(dataset_dir, dataset_name)
            self.plot_raw_population_fr(dataset_dir, dataset_name)
            self.plot_synchrony_index_over_time(dataset_dir, dataset_name)
            self.plot_active_units_per_electrode(dataset_dir, dataset_name)
            self.plot_electrode_activity_heatmap(dataset_dir, dataset_name)
            self.plot_sttc_over_time(dataset_dir, dataset_name)
            self.plot_footprint_sttc(dataset_dir, dataset_name)
            self.plot_sttc_log(dataset_dir, dataset_name)
            self.plot_sttc_vmin_vmax(dataset_dir, dataset_name)
            self.plot_sttc_thresh(dataset_dir, dataset_name)
            self.plot_kde_pdf(dataset_dir, dataset_name)
            self.sttc_violin_plot_by_firing_rate(dataset_dir, [dataset_name])
            self.sttc_violin_plot_by_proximity(dataset_dir, [dataset_name])


        # generate comparison plots
        self.sttc_violin_plot_by_firing_rate(comparison_dir, base_names)
        self.sttc_violin_plot_by_proximity(comparison_dir, base_names)
        self.sttc_violin_plot_across_recordings(comparison_dir, base_names)
        self.plot_comparison_inverse_isi(comparison_dir, base_names)
        self.plot_comparison_regular_isi(comparison_dir, base_names)
        self.plot_comparison_firing_rate_histogram(comparison_dir, base_names)
        self.plot_comparison_kde_pdf(comparison_dir, base_names)
        self.analyze_burst_characteristics(comparison_dir, base_names)




        #linear comparisons
        metrics = ["sttc", "firing_rate", "isi"]
        for metric in metrics:
            self.plot_pairwise_linear_comparison(pairwise_dir, base_names, metric)

        #global linear comparison (for 3+ datasets)
        if len(self.trains) >= 3:
            for metric in metrics:
                self.plot_global_linear_comparison(global_dir, base_names, metric)


        # optional PCA analysis
        if perform_pca:
            # prepare PCA
            pca_result, explained_variance = self.prepare_pca(self.firing_rates_list) #averaged pca
            pca_continuous_result, explained_variance_continuous, time_bins_continuous = self.calculate_continuous_pca() #continuous pca

            # plot  pca
            self.plot_pca(pca_result, explained_variance, comparison_dir, base_names) #averaged pca
            self.plot_continuous_pca(pca_continuous_result, explained_variance_continuous, time_bins_continuous, comparison_dir, base_names) #continuous pca


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


# main function
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

    # generate base names from input files for naming output zip file
    base_names = [os.path.splitext(os.path.basename(path))[0].replace('_acqm', '') for path in input_s3]
    combined_name = "_".join(base_names[:2])  # Use first two names for naming

    # create local temporary directories for processing
    local_output_folder = f'/tmp/output_plots_{combined_name}'

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
    analysis = SpikeDataAnalysis(local_input_paths)
    zip_filename = analysis.run_all_analyses(local_output_folder, base_names, perform_pca=perform_pca, cleanup=cleanup)

    output_s3 = os.path.join(output_s3, os.path.basename(zip_filename))
    # upload zip to S3
    print(f"Uploading {zip_filename} to {output_s3}")
    wr.upload(zip_filename, output_s3)

    print("Analysis complete. Results uploaded to S3.")




# main
if __name__ == '__main__':
    main()

