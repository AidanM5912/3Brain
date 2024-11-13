import argparse
import braingeneers.utils.s3wrangler as wr
from datetime import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
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
            plt.savefig(os.path.join(output_path, f"raster_plot_{dataset_name}.png"))
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

    def get_population_fr(self, train, bin_size=0.1, w=5):
        trains = np.hstack(train)
        rec_length = np.max(trains)
        bin_num = int(rec_length // bin_size) + 1
        bins = np.linspace(0, rec_length, bin_num)
        fr = np.histogram(trains, bins)[0] / bin_size
        fr_avg = np.convolve(fr, np.ones(w), 'same') / w
        return bins[1:], fr_avg

    def population_fr_plot(self, output_path, dataset_name):
        plt.close('all')
        for i, train in enumerate(self.trains):

            bins, fr_avg = self.get_population_fr(train)

            plt.figure(figsize=(12, 6))
            plt.plot(bins, fr_avg)

            plt.xlabel("Time (s)", fontsize=12)
            plt.ylabel("Population Firing Rate (Hz)", fontsize=12)

            plt.xlim(0, self.durations[i])
            plt.ylim(np.min(fr_avg) - 5, np.max(fr_avg) + 5)

            plt.title(f"Population Firing Rate: {dataset_name}", fontsize=16)
            plt.savefig(os.path.join(output_path, f"population_fr_plot_{dataset_name}.png"))
            plt.close()

    def overlay_fr_raster(self, output_path, dataset_name):
        plt.close('all')
        for i, train in enumerate(self.trains):

            bins, fr_avg = self.get_population_fr(train)

            fig, axs = plt.subplots(1, 1, figsize=(16, 6))
            axs1 = axs.twinx()

            axs1.plot(bins, fr_avg, color='r', linewidth=3, alpha=0.5)

            y = 0
            for vv in train:
                axs.scatter(vv, [y] * len(vv), marker="|", c='k', s=4, alpha=0.7)
                y += 1

            axs.set_xlabel("Time (s)", fontsize=14)
            axs.set_ylabel("Neuron Number", fontsize=14)
            axs1.set_ylabel("Population Firing Rate (Hz)", fontsize=16, color='r')
            axs1.spines['right'].set_color('r')
            axs1.spines['right'].set_linewidth(2)
            axs1.tick_params(axis='y', colors='r')

            axs.set_title(f"Population Level Activity: {dataset_name}", fontsize=16)
            plt.savefig(os.path.join(output_path, f"overlay_fr_raster_{dataset_name}.png"))
            plt.close(fig)

    def compute_sttc_matrix(self, spike_train, length, delt=20):
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


    def get_upper_triangle_values(self, matrix):
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

    def group_neurons_by_firing_rate(self, firing_rates):
        low_threshold = np.percentile(firing_rates, 33)
        high_threshold = np.percentile(firing_rates, 66)

        groups = {
            'low': [i for i, rate in enumerate(firing_rates) if rate < low_threshold],
            'medium': [i for i, rate in enumerate(firing_rates) if low_threshold <= rate < high_threshold],
            'high': [i for i, rate in enumerate(firing_rates) if rate >= high_threshold]
        }
        return groups

    def group_neurons_by_proximity(self, neuron_data, distance_threshold=100):
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

    def sttc_violin_plot_by_firing_rate(self, output_path, dataset_names):
        plt.close('all')
        # individual plots for each dataset
        for i, train in enumerate(self.trains):
            dataset_name = dataset_names[i]
            groups = self.group_neurons_by_firing_rate(self.firing_rates_list[i])
            sttc_values = []
            labels = []

            for group_name, indices in groups.items():
                sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                labels.append(f"{group_name.capitalize()}")

            # create the individual plot for the current dataset
            plt.figure(figsize=(12, 8))
            plt.violinplot(sttc_values)
            plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha='right')
            plt.xlabel('Firing Rate Group')
            plt.ylabel('STTC Values')
            plt.title(f'Violin Plot of STTC Values by Firing Rate: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_firing_rate_{dataset_name}.png"))
            plt.close()

        # combined plot for all datasets
        combined_sttc_values = []
        combined_labels = []
        plt.close('all')
        for i, train in enumerate(self.trains):
            dataset_name=dataset_names[i]
            groups = self.group_neurons_by_firing_rate(self.firing_rates_list[i])

            for group_name, indices in groups.items():
                sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                combined_sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                combined_labels.append(f"{dataset_name} - {group_name.capitalize()}")

        plt.figure(figsize=(12, 8))
        plt.violinplot(combined_sttc_values)
        plt.xticks(ticks=np.arange(1, len(combined_labels) + 1), labels=combined_labels, rotation=45, ha='right')
        plt.xlabel('Firing Rate Group')
        plt.ylabel('STTC Values')
        plt.title('Violin Plot of STTC Values by Firing Rate Group Across Recordings')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_firing_rate_combined.png"))
        plt.close()



    def sttc_violin_plot_by_proximity(self, output_path, dataset_names, distance_threshold=100):
    # individual plots for each dataset
        plt.close('all')
        for i, train in enumerate(self.trains):
            dataset_name=dataset_names[i]
            groups = self.group_neurons_by_proximity(self.neuron_data_list[i], distance_threshold)
            sttc_values = []
            labels = []

            for group_name, indices in groups.items():
                if len(indices) < 2:
                    continue
                sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                labels.append(f"{group_name.capitalize()}")

            # create the individual plot for the current dataset
            plt.figure(figsize=(12, 8))
            plt.violinplot(sttc_values)
            plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha='right')
            plt.xlabel('Proximity Group')
            plt.ylabel('STTC Values')
            plt.title(f'Violin Plot of STTC Values by Spatial Proximity: {dataset_name} (Threshold = {distance_threshold}μm)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_proximity_{dataset_name}.png"))
            plt.close()

        # combined plot for all datasets
        plt.close('all')
        combined_sttc_values = []
        combined_labels = []
        for i, train in enumerate(self.trains):
            dataset_name=dataset_names[i]
            groups = self.group_neurons_by_proximity(self.neuron_data_list[i], distance_threshold)

            for group_name, indices in groups.items():
                if len(indices) < 2:
                    continue
                sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                combined_sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                combined_labels.append(f"{dataset_name} - {group_name.capitalize()}")

        plt.figure(figsize=(12, 8))
        plt.violinplot(combined_sttc_values)
        plt.xticks(ticks=np.arange(1, len(combined_labels) + 1), labels=combined_labels, rotation=45, ha='right')
        plt.xlabel('Proximity Group')
        plt.ylabel('STTC Values')
        plt.title(f'Violin Plot of STTC Values by Spatial Proximity Across Recordings (Threshold = {distance_threshold}μm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_proximity_combined.png"))
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
        plt.violinplot(sttc_values_list)
        plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha ='right')
        plt.xlabel('Recordings')
        plt.ylabel('STTC Values')
        plt.title('Violin Plot of STTC Values Across Recordings')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_across_recordings.png"))
        plt.close()

    def plot_inter_burst_interval_distribution(self, output_path, dataset_name):
        plt.close('all')
        for i, train in enumerate(self.trains):
            burst_intervals = []
            for neuron_spikes in train:
                burst_intervals.extend(np.diff(neuron_spikes))  # calculate intervals between bursts

            plt.figure(figsize=(12, 6))
            plt.hist(burst_intervals, bins=50, color='blue', alpha=0.7)
            plt.xlabel('Inter-Burst Interval (s)')
            plt.ylabel('Count')
            plt.title(f'Inter-Burst Interval Distribution: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"inter_burst_interval_distribution_{dataset_name}.png"))
            plt.close()

    def plot_burst_frequency_duration(self, output_path, dataset_name):
        plt.close('all')
        for i, train in enumerate(self.trains):
            burst_frequencies = []
            burst_durations = []

            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)
                if intervals.size > 0:
                    burst_frequencies.append(1 / np.mean(intervals))  # frequency as the inverse of the mean interval
                    burst_durations.append(np.sum(intervals))        # total duration of bursts

            plt.figure(figsize=(12, 6))
            plt.scatter(burst_frequencies, burst_durations, alpha=0.6)
            plt.xlabel('Burst Frequency (Hz)')
            plt.ylabel('Burst Duration (s)')
            plt.title(f'Burst Frequency vs. Duration: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"burst_frequency_duration_{dataset_name}.png"))
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

    def plot_population_firing_rate_over_time(self, output_path, dataset_name, bin_size=1.0):
        plt.close('all')
        for i, train in enumerate(self.trains):
            combined_train = np.hstack(train)
            max_time = np.max(combined_train)
            bins = np.arange(0, max_time + bin_size, bin_size)
            firing_rate, _ = np.histogram(combined_train, bins=bins)

            plt.figure(figsize=(12, 6))
            plt.plot(bins[:-1], firing_rate, color='red')
            plt.xlabel('Time (s)')
            plt.ylabel('Population Firing Rate (Hz)')
            plt.title(f'Population Firing Rate Over Time: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"population_firing_rate_over_time_{dataset_name}.png"))
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

            sttc_sums = np.sum(sttc_matrix[i])-1 #sum an indiviudal neurons sttc values minus the interaction with itself

            for j, neuron in enumerate(neuron_data.values()):
                x, y = neuron['position']
                if (x, y) != (0, 0):  # exclude the calibration electrode
                    neuron_x.append(x)
                    neuron_y.append(y)
                    sttc_marker_size.append(sttc_sums[j])

            legend_rates = np.percentile([sttc_sums > 0], [50, 75, 90, 98])

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



    def run_all_analyses(self, output_folder, base_names, cleanup=True):
        os.makedirs(output_folder, exist_ok=True)# name the folder based on input file names

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        dataset_directories = []
        for base_name in base_names:
            dataset_dir = os.path.join(output_folder, base_name)
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_directories.append((dataset_dir, base_name))


        # create a directory for comparison plots
        comparison_dir = os.path.join(output_folder, "comparisons")
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)

        # list of plotting methods for each dataset for processing
        for i, (dataset_dir, dataset_name) in enumerate(dataset_directories):
            self.raster_plot(dataset_dir, dataset_name)
            self.footprint_opaque_circles(dataset_dir, dataset_name)
            self.overlay_fr_raster(dataset_dir, dataset_name)
            self.population_fr_plot(dataset_dir, dataset_name)
            self.sttc_plot(dataset_dir, dataset_name)
            self.plot_inter_burst_interval_distribution(dataset_dir, dataset_name)
            self.plot_burst_frequency_duration(dataset_dir, dataset_name)
            self.plot_firing_rate_histogram(dataset_dir, dataset_name)
            self.plot_firing_rate_cdf(dataset_dir, dataset_name)
            self.plot_isi_histogram(dataset_dir, dataset_name)
            self.plot_cv_of_isi(dataset_dir, dataset_name)
            self.plot_population_firing_rate_over_time(dataset_dir, dataset_name)
            self.plot_synchrony_index_over_time(dataset_dir, dataset_name)
            self.plot_active_units_per_electrode(dataset_dir, dataset_name)
            self.plot_electrode_activity_heatmap(dataset_dir, dataset_name)
            self.plot_sttc_over_time(dataset_dir, dataset_name)
            self.plot_footprint_sttc(dataset_dir, dataset_name)

            gc.collect()

        # generate comparison plots sequentially
        self.sttc_violin_plot_by_firing_rate(comparison_dir, base_names)
        self.sttc_violin_plot_by_proximity(comparison_dir, base_names)
        self.sttc_violin_plot_across_recordings(comparison_dir, base_names)

        gc.collect() #free up memory

        # zip the local output folder
        zip_filename = f"{output_folder}.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            for foldername, subfolders, filenames in os.walk(output_folder):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    zip_file.write(file_path, os.path.relpath(file_path, output_folder))

        # clean up the output folder after zipping (optional)
        if cleanup:
            shutil.rmtree(output_folder)

        return zip_filename



def main():
    parser = argparse.ArgumentParser(description="Plotting script for the auto-curation output (a zipped numpy array).")
    # positional argument for input paths (S3 paths)
    parser.add_argument("input_s3", nargs='+', help="Input file paths (S3 paths)")
    # positional argument for the output path (S3)
    parser.add_argument("output_path", help="Output S3 path for the zip file")
    # optional flag for cleanup
    parser.add_argument("--cleanup", action="store_true", help="Delete the output folder after zipping")
    args = parser.parse_args()

    # access the parsed arguments, defining them as input and output
    input_s3 = args.input_s3
    output_s3 = args.output_path
    cleanup = args.cleanup

     # generate the base names from input files for naming the output folder and ZIP file
    base_names = [os.path.splitext(os.path.basename(path))[0].replace('_acqm', '') for path in input_s3]
    combined_name = "_".join(base_names[:2])  # use the first two names if there are multiple files

    # create a temporary local directory for processing
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
    zip_filename = analysis.run_all_analyses(local_output_folder, base_names, cleanup=cleanup)

    output_s3 = os.path.join(args.output_path, os.path.basename(zip_filename))
    # upload the zipped file back to S3
    print(f"Uploading {zip_filename} to {output_s3}")
    wr.upload(zip_filename, output_s3)

    print("Plotting complete, zip uploaded to S3.")


# main
if __name__ == '__main__':
    main()

