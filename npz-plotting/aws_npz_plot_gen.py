import argparse
import braingeneers.utils.s3wrangler as wr
from datetime import datetime
import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import re
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
        
        #generate original and cleaned names
        self.original_names = [os.path.splitext(os.path.basename(path))[0] for path in self.input_paths]
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


    def plot_smoothed_population_fr(self, output_path, dataset_name):
        print(f"Plotting smoothed population firing rate for {dataset_name}")
        plt.close('all')
        for i, train in enumerate(self.trains):

            bins, fr_avg = self.get_population_fr(train, smoothing=True)

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

            bins, fr_avg = self.get_population_fr(train, smoothing=True)

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

    def compute_spike_triggered_sttc(self, trains, duration, window_size=1.0):
        """
        Optimized spike-triggered STTC computation. For use in large datasets (NOT IMPLEMENTED YET)

        Parameters:
            trains (list of arrays): Spike trains, each array corresponds to a neuron's spike times.
            duration (float): Duration of the recording in seconds.
            window_size (float): Time window size for the STTC calculation.

        Returns:
            numpy.ndarray: Pairwise STTC matrix.
        """
        import scipy.spatial

        num_neurons = len(trains)
        sttc_matrix = np.zeros((num_neurons, num_neurons))
        tau = window_size / duration  # Precompute tau as it is constant

        # Precompute pairwise spike distances for all neuron pairs
        for i in range(num_neurons):
            train1 = trains[i]

            for j in range(i + 1, num_neurons):  # Upper triangular matrix
                train2 = trains[j]

                # Efficient pairwise spike distance computation using KDTree
                tree1 = scipy.spatial.cKDTree(train1[:, None])  # Build KDTree for fast lookups
                tree2 = scipy.spatial.cKDTree(train2[:, None])

                # Find neighbors within the window size for both trains
                prop1 = len(tree1.query_ball_tree(tree2, r=window_size)) / len(train2)
                prop2 = len(tree2.query_ball_tree(tree1, r=window_size)) / len(train1)

                # Compute STTC
                sttc = 0.5 * ((prop1 - tau) / (1 - tau) + (prop2 - tau) / (1 - tau))
                sttc_matrix[i, j] = sttc
                sttc_matrix[j, i] = sttc  # Symmetric

        return sttc_matrix

    @staticmethod
    def group_neurons_by_firing_rate(firing_rates):
        low_threshold = np.percentile(firing_rates, 33)
        high_threshold = np.percentile(firing_rates, 66)

        groups = {
            'low': [i for i, rate in enumerate(firing_rates) if rate < low_threshold],
            'medium': [i for i, rate in enumerate(firing_rates) if low_threshold <= rate < high_threshold],
            'high': [i for i, rate in enumerate(firing_rates) if rate >= high_threshold]
        }
        return groups, low_threshold, high_threshold

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
        print(f"Plotting STTC matrix for {dataset_name}")
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

    def sttc_violin_plot_by_firing_rate_individual(self, output_path, dataset_name):
        #generate STTC violin plots for an individual dataset by firing rate.
        print(f"Generating STTC violin plots by firing rate for {dataset_name}")
        plt.close('all')

        for i, train in enumerate(self.trains):
            groups, low_threshold, high_threshold = self.group_neurons_by_firing_rate(self.firing_rates_list[i])
            sttc_values = []
            labels = []

            for group_name, indices in groups.items():
                sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                if group_name == 'low':
                    labels.append(f"Low (<{low_threshold:.2f} Hz)")
                elif group_name == 'medium':
                    labels.append(f"Medium ({low_threshold:.2f}-{high_threshold:.2f} Hz)")
                elif group_name == 'high':
                    labels.append(f"High (≥{high_threshold:.2f} Hz)")


        # do the violin plot
        plt.figure(figsize=(12, 8))
        plt.violinplot(sttc_values, showmeans=True)
        plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha='right')
        plt.xlabel('Firing Rate Group')
        plt.ylabel('STTC Values')
        plt.title(f'STTC Violin Plot by Firing Rate: {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_firing_rate_{dataset_name}.png"))
        plt.close()

    def sttc_violin_plot_by_firing_rate_compare(self, output_path, dataset_names):
        #generate a combined STTC violin plot by firing rate across multiple datasets.
        print(f"Generating combined STTC violin plots by firing rate for {dataset_names}")
        plt.close('all')

        combined_sttc_values = []
        combined_labels = []

        for i, train in enumerate(self.trains):
            dataset_name = dataset_names[i]
            groups, low_threshold, high_threshold = self.group_neurons_by_firing_rate(self.firing_rates_list[i])

            for group_name, indices in groups.items():
                sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                combined_sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                if group_name == 'low':
                    combined_labels.append(f"{dataset_name} - Low (<{low_threshold:.2f} Hz)")
                elif group_name == 'medium':
                    combined_labels.append(f"{dataset_name} - Medium ({low_threshold:.2f}-{high_threshold:.2f} Hz)")
                elif group_name == 'high':
                    combined_labels.append(f"{dataset_name} - High (≥{high_threshold:.2f} Hz)")



        # Generate the violin plot
        plt.figure(figsize=(12, 8))
        plt.violinplot(combined_sttc_values, showmeans=True)
        plt.xticks(ticks=np.arange(1, len(combined_labels) + 1), labels=combined_labels, rotation=45, ha='right')
        plt.xlabel('Firing Rate Group')
        plt.ylabel('STTC Values')
        plt.title('STTC Violin Plot by Firing Rate Group Across Recordings')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_firing_rate_combined.png"))
        plt.close()

    def sttc_violin_plot_by_proximity_individual(self, output_path, dataset_name, distance_threshold=100):
        """
        Generate STTC violin plots for an individual dataset by proximity.

        Parameters:
            output_path (str): Directory to save plots.
            dataset_name (str): Name of the dataset.
            distance_threshold (int): Threshold for proximity grouping.
        """
        print(f"Generating STTC violin plots by proximity for {dataset_name}")
        plt.close('all')

        # Find the index of the dataset
        for i, train in enumerate(self.trains):
            neuron_data = self.neuron_data_list[i]
            groups = self.group_neurons_by_proximity(neuron_data, distance_threshold)
            sttc_values = []
            labels = []

            for group_name, indices in groups.items():
                if len(indices) < 2:
                    continue
                sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                labels.append(f"{group_name.capitalize()}")

        # Generate the violin plot
        plt.figure(figsize=(12, 8))
        plt.violinplot(sttc_values, showmeans=True)
        plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha='right')
        plt.xlabel('Proximity Group')
        plt.ylabel('STTC Values')
        plt.title(f'STTC Violin Plot by Spatial Proximity: {dataset_name} (Proximity ≤ {distance_threshold} μm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_proximity_{dataset_name}.png"))
        plt.close()

    def sttc_violin_plot_by_proximity_compare(self, output_path, dataset_names, distance_threshold=100):
        """
        Generate a combined STTC violin plot by proximity across multiple datasets.

        Parameters:
            output_path (str): Directory to save plots.
            dataset_names (list): Names of datasets for comparison.
            distance_threshold (int): Threshold for proximity grouping.
        """
        print(f"Generating combined STTC violin plots by proximity for {dataset_names}")
        plt.close('all')

        combined_sttc_values = []
        combined_labels = []

        for i, train in enumerate(self.trains):
            dataset_name = dataset_names[i]
            neuron_data = self.neuron_data_list[i]
            groups = self.group_neurons_by_proximity(neuron_data, distance_threshold)

            for group_name, indices in groups.items():
                if len(indices) < 2:
                    continue
                sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                combined_sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                combined_labels.append(f"{dataset_name} - {group_name.capitalize()}")

        # Generate the violin plot
        plt.figure(figsize=(12, 8))
        plt.violinplot(combined_sttc_values, showmeans=True)
        plt.xticks(ticks=np.arange(1, len(combined_labels) + 1), labels=combined_labels, rotation=45, ha='right')
        plt.xlabel('Proximity Group')
        plt.ylabel('STTC Values')
        plt.title(f'STTC Violin Plot by Spatial Proximity Across Recordings (Proximity ≤ {distance_threshold}μm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_proximity_combined.png"))
        plt.close()


    def sttc_violin_plot_across_recordings(self, output_path, dataset_names):
        print("Generating STTC violin plot across recordings")
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
        print(f"Plotting firing rate histogram for {dataset_name}")
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
        print(f"Plotting firing rate CDF for {dataset_name}")
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

    def plot_isi_histogram(self, output_path, dataset_name, bins=50, log_scale=False, xlim=None, kde=False):
        print(f"Plotting ISI histogram for {dataset_name}")
        """
        Plot the ISI histogram for a dataset with enhancements.

        Parameters:
            output_path (str): Directory to save the plot.
            dataset_name (str): Name of the dataset.
            bins (int or array): Number of bins or bin edges for the histogram.
            log_scale (bool): If True, use a logarithmic x-axis.
            xlim (tuple): Limits for the x-axis (e.g., (0, 8) to focus on short ISIs).
            kde (bool): If True, overlay a Kernel Density Estimation (KDE).
        """
        plt.close('all')

        for i, train in enumerate(self.trains):
            # Collect all ISI values
            all_intervals = []
            for neuron_spikes in train:
                all_intervals.extend(np.diff(neuron_spikes))

            # Filter ISI values based on xlim
            if xlim:
                all_intervals = [isi for isi in all_intervals if xlim[0] <= isi <= xlim[1]]

            # Create histogram
            plt.figure(figsize=(12, 6))
            if log_scale:
                # Use logarithmic bins if log_scale is True
                bins = np.logspace(np.log10(min(all_intervals) + 1e-6), np.log10(max(all_intervals)), bins)
                plt.xscale('log')
            else:
                # Use linear bins (adjusted to xlim if specified)
                bins = np.linspace(xlim[0], xlim[1], bins) if xlim else bins

            plt.hist(all_intervals, bins=bins, color='red', alpha=0.7, label='Histogram')

            # Add KDE if enabled
            if kde:
                from scipy.stats import gaussian_kde
                density = gaussian_kde(all_intervals)
                xs = np.linspace(min(all_intervals), max(all_intervals), 500)
                plt.plot(xs, density(xs), color='blue', label='KDE')

            # Set x-axis limits
            if xlim:
                plt.xlim(xlim)  # Explicitly set the x-axis range

            # Labels and title
            plt.xlabel('Inter-Spike Interval (s)')
            plt.ylabel('Count')
            plt.title(f'ISI Histogram: {dataset_name}')
            plt.legend()
            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(output_path, f"isi_histogram_{dataset_name}_log_{log_scale}.png"))
            plt.close()



    def plot_cv_of_isi(self, output_path, dataset_name, bins=50, xlim=None, kde=False):
        print(f"Plotting Coefficient of Variation of ISI for {dataset_name}")
        """
        Plot the histogram of the Coefficient of Variation (CV) of ISI values.

        Parameters:
            output_path (str): Directory to save the plot.
            dataset_name (str): Name of the dataset.
            bins (int or array): Number of bins or bin edges for the histogram.
            xlim (tuple): Limits for the x-axis (e.g., (0, 2) to focus on low CV values).
            kde (bool): If True, overlay a Kernel Density Estimation (KDE).
        """
        plt.close('all')

        for i, train in enumerate(self.trains):
            # Calculate CV values
            cv_values = []
            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)
                if len(intervals) > 1:
                    cv = np.std(intervals) / np.mean(intervals)  # CV: std/mean
                    cv_values.append(cv)

            # Filter CV values based on xlim
            if xlim:
                cv_values = [cv for cv in cv_values if xlim[0] <= cv <= xlim[1]]

            # Create histogram
            plt.figure(figsize=(12, 6))
            bins = np.linspace(xlim[0], xlim[1], bins) if xlim else bins  # Adjust bins to range
            plt.hist(cv_values, bins=bins, color='orange', alpha=0.7, label='Histogram')

            # Add KDE if enabled
            if kde:
                from scipy.stats import gaussian_kde
                density = gaussian_kde(cv_values)
                xs = np.linspace(min(cv_values), max(cv_values), 500)
                plt.plot(xs, density(xs), color='blue', label='KDE')

            # Set x-axis limits if specified
            if xlim:
                plt.xlim(xlim)

            # Labels and title
            plt.xlabel('Coefficient of Variation (CV)')
            plt.ylabel('Count')
            plt.title(f'Coefficient of Variation of ISI: {dataset_name}')
            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(output_path, f"cv_of_isi_{dataset_name}.png"))
            plt.close()

    def plot_raw_population_fr(self, output_path, dataset_name, bin_size=1.0):
        print(f"Plotting raw population firing rate for {dataset_name}")
        plt.close('all')
        for i, train in enumerate(self.trains):
            combined_train = np.hstack(train)
            max_time = np.max(combined_train)
            bins = np.arange(0, max_time + bin_size, bin_size)
            firing_rate, _ = np.histogram(combined_train, bins=bins)


             # Normalize by the number of neurons
            num_neurons = len(train)
            if num_neurons > 0:
                firing_rate = firing_rate / num_neurons  # Normalize firing rate per neuron


            plt.figure(figsize=(12, 6))
            plt.plot(bins[:-1], firing_rate, color='red')
            plt.xlabel('Time (s)')
            plt.ylabel('Raw Population Firing Rate (Hz)')
            plt.title(f'Raw Population Firing Rate: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"raw_population_fr_{dataset_name}.png"))
            plt.close()

    def plot_synchrony_index_over_time(self, output_path, dataset_name, window_size=10.0):
        print(f"Plotting synchrony index over time for {dataset_name}")
        plt.close('all')
        for i, train in enumerate(self.trains):
            synchrony_indices = []
            time_points = []
            max_time = np.max(np.hstack(train))
            current_time = 0

            while current_time + window_size <= max_time:
                # Extract spikes within the current time window
                windowed_trains = [
                    spike_times[(spike_times >= current_time) & (spike_times < current_time + window_size)] 
                    for spike_times in train
                ]
                
                # Pad arrays to the same length
                max_len = max(len(trains) for trains in windowed_trains)
                padded_trains = np.array([np.pad(trains, (0, max_len - len(trains)), 'constant') for trains in windowed_trains])

                # Skip if all padded arrays are zeros or contain only a single unique value
                if np.all(padded_trains == 0) or np.all(padded_trains == padded_trains[0]):
                    current_time += window_size
                    continue
                
                # Remove rows with zero standard deviation
                valid_indices = [j for j in range(padded_trains.shape[0]) if np.std(padded_trains[j]) > 0]
                if len(valid_indices) > 1:
                    filtered_trains = padded_trains[valid_indices]
                    
                    # Compute pairwise correlations
                    pairwise_corr = np.corrcoef(filtered_trains)
                    if not np.isnan(pairwise_corr).any():  # Skip if correlation contains NaNs
                        synchrony_indices.append(np.mean(pairwise_corr[np.triu_indices_from(pairwise_corr, k=1)]))
                        time_points.append(current_time + window_size / 2)

                current_time += window_size

            if synchrony_indices:
                plt.figure(figsize=(12, 6))
                plt.plot(time_points, synchrony_indices, color='magenta')
                plt.xlabel('Time (s)')
                plt.ylabel('Synchrony Index')
                plt.title(f'Synchrony Index Over Time (Window Size={window_size:.1f}s): {dataset_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"synchrony_index_over_time_{dataset_name}.png"))
                plt.close()
            else:
                print(f"No valid synchrony data for plotting for dataset: {dataset_name}")


    def plot_active_units_per_electrode(self, output_path, dataset_name):
        print(f"Plotting active units per electrode for {dataset_name}")
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
        print(f"Plotting electrode activity heatmap for {dataset_name}")
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
        print(f"Plotting STTC over time for {dataset_name}")
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
            plt.title(f'STTC Over Time (Window Size={window_size:.1f}s): {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"sttc_over_time_{dataset_name}.png"))
            plt.close()
    
    def plot_footprint_sttc(self, output_path, dataset_name):
        print(f"Plotting footprint plot with STTC for {dataset_name}")
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

            neuron_x = np.array(neuron_x)
            neuron_y = np.array(neuron_y)
            sttc_marker_size = np.array(sttc_marker_size)

            legend_rates = np.percentile(sttc_sums, [50, 75, 90, 98])

            plt.figure(figsize=(11, 9))
            plt.scatter(neuron_x, neuron_y, s=sttc_marker_size * 100, alpha=0.4, c='r', edgecolors='none')
        
            for rate in legend_rates:
                plt.scatter([], [], s=rate * 100, c='r', alpha=0.4, label=f'STTC Sum {rate /100:.2f}')

            plt.legend(scatterpoints=1, frameon=True, labelspacing=1.4, handletextpad=0.8, borderpad=0.92, title='STTC Sum', loc = 'best', title_fontsize=10, fontsize=10, )
            plt.xlabel(r"Horizontal Position ($\mu$m)")
            plt.ylabel(r"Vertical Position ($\mu$m)")
            plt.title(f"Footprint Plot with STTC Sum : {dataset_name}")
            plt.savefig(os.path.join(output_path, f"footprint_plot_sttc_sum_{dataset_name}.png"))
            plt.close()

    def plot_comparison_inverse_isi(self, output_path, base_names):
        """
        Generate and save a comparison overlay plot for population-level inverse ISI for all datasets,
        normalized by the number of neurons.
        """
        print("Generating comparison overlay plot for inverse ISI")
        plt.figure(figsize=(12, 6))
        for i, train in enumerate(self.trains):
            all_intervals = []
            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)
                if len(intervals) > 0:
                    all_intervals.extend(intervals)

            # Remove zero intervals before calculating inverse ISI
            all_intervals = np.array(all_intervals)
            all_intervals = all_intervals[all_intervals > 0]  # Exclude zero or negative intervals

            if len(all_intervals) > 0:
                inverse_isi = 1 / all_intervals
                inverse_isi = inverse_isi[np.isfinite(inverse_isi)]  # Exclude any infinities
                
                # Normalize by the number of neurons in the dataset
                num_neurons = len(train)
                if num_neurons > 0:
                    inverse_isi /= num_neurons  # Normalize firing rate

                # Apply clipping to remove extreme outliers
                inverse_isi = inverse_isi[inverse_isi < 1000]  # Exclude values above 1000 Hz

                # Use logarithmic bins if appropriate
                bins = np.logspace(np.log10(1), np.log10(max(inverse_isi)), 50)  # Logarithmic bins
                plt.hist(inverse_isi, bins=bins, alpha=0.5, label=base_names[i], density=True)
                plt.xscale('log')  # Set x-axis to logarithmic scale

        plt.xlabel("Normalized Instantaneous Firing Rate (Hz/neuron)")
        plt.ylabel("Density")
        plt.title("Comparison: Population-Level Inverse ISI Across Datasets (Normalized)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "comparison_population_inverse_isi_normalized.png"))
        plt.close()


    def plot_comparison_regular_isi(self, output_path, base_names):
        """
        Generate and save a comparison overlay plot for regular ISI histograms for all datasets.
        """
        print("Generating comparison overlay plot for regular ISI")
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
        print(f"Finding neuron of interest for dataset {dataset_index}")
        if dataset_index >= len(self.trains):
            print(f"Dataset index {dataset_index} out of bounds.")
            print(f"Dataset index {dataset_index} out of bounds. Total datasets: {len(self.trains)}")
            return None
        print(f"Dataset {dataset_index} has {len(self.trains[dataset_index])} neurons and duration {self.durations[dataset_index]}.")

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
        print(f"Plotting population-level inverse ISI for {dataset_name}")
        for i, train in enumerate(self.trains):
            all_intervals = []
            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)
                if len(intervals) > 0:
                    all_intervals.extend(intervals)

            # Convert to a NumPy array and filter out zero or negative intervals
            all_intervals = np.array(all_intervals)
            all_intervals = all_intervals[all_intervals > 0]  # Exclude zero or negative intervals

            if len(all_intervals) > 0:
                # Calculate inverse ISI and exclude infinities
                inverse_isi = 1 / all_intervals
                inverse_isi = inverse_isi[np.isfinite(inverse_isi)]  # Exclude infinities

                # Normalize by the number of neurons
                num_neurons = len(train)
                if num_neurons > 0:
                    inverse_isi /= num_neurons

                # Clip extreme values to avoid distortion in the plot
                inverse_isi = inverse_isi[inverse_isi < 500]  # Adjust threshold as needed

                # Create linear bins for the histogram
                bins = np.linspace(0, 20, 50)  # Adjust range and bin count based on data

                # Plot the histogram
                plt.figure(figsize=(12, 6))
                plt.hist(inverse_isi, bins=bins, color='green', alpha=0.7, density=True)
                plt.xlabel("Normalized Instantaneous Firing Rate (Hz/neuron)")
                plt.ylabel("Density")
                plt.title(f"Population-Level Inverse ISI: {dataset_name}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f"population_inverse_isi_{dataset_name}.png"))
                plt.close()


    def plot_neuron_inverse_isi(self, output_path, dataset_name, dataset_index=0):
        """
        Plot the inverse ISI histogram for a single neuron of interest.

        Parameters:
            output_path (str): Directory to save the plot.
            dataset_name (str): Name of the dataset.
            dataset_index (int): Index of the dataset to analyze.
        """
        print(f"Plotting inverse ISI for neuron of interest in dataset {dataset_name}")
        # find the neuron of interest
        neuron_index = self.find_neuron_of_interest(dataset_index)
        if neuron_index is None:
            print(f"Could not determine neuron of interest for dataset {dataset_name}. Skipping.")
            return

        # get the spike train for the specified neuron
        train = self.trains[dataset_index]
        intervals = np.diff(train[neuron_index])  # compute ISI

        # filter invalid intervals
        intervals = intervals[intervals > 0]  # remove zero or negative intervals

        if len(intervals) > 0:
            # calculate inverse ISI and filter invalid values
            inverse_isi = 1 / intervals
            inverse_isi = inverse_isi[np.isfinite(inverse_isi)]  # remove infinities

            # clip extreme values for better visualization
            inverse_isi = inverse_isi[inverse_isi < 500]  # exclude values above 500 Hz

            # plot the histogram
            bins = np.linspace(0, 100, 50)  # linear bins, adjust range as needed
            plt.figure(figsize=(12, 6))
            plt.hist(inverse_isi, bins=bins, color='blue', alpha=0.7, density=True)
            plt.xlabel("Instantaneous Firing Rate (Hz)")
            plt.ylabel("Density")
            plt.title(f"Neuron {neuron_index} Inverse ISI: {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"neuron_{neuron_index}_inverse_isi_{dataset_name}.png"))
            plt.close()
        else:
            print(f"No valid ISI intervals found for neuron {neuron_index} in dataset {dataset_name}. Skipping.")


    def plot_neuron_regular_isi(self, output_path, dataset_name, dataset_index=0):
        print(f"Plotting regular ISI for neuron of interest in dataset {dataset_name}")
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

    def plot_high_activity_rasters(self, output_path, dataset_name):
        """
        Generate raster plots for high-activity periods with smaller time windows.

        Parameters:
            output_path (str): Directory to save the raster plots.
            dataset_name (str): Name of the dataset.

        Saves:
            Raster plots for specified time windows centered on high-activity periods.
        """
        print(f"Generating high-activity rasters for {dataset_name}")
        for i, train in enumerate(self.trains):
            # calculate population firing rate
            bins, fr_avg = self.get_population_fr(train, smoothing=True)

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
        print("Generating comparison overlay plot for firing rate histograms")

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
        print(f"Plotting STTC heatmap on a logarithmic scale for {dataset_name}")
        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            sttc_matrix = self.compute_sttc_matrix(train, duration)

            plt.figure(figsize=(10, 8))
            log_norm = LogNorm(vmin=max(np.min(sttc_matrix), 1e-6), vmax=np.max(sttc_matrix))
            plt.imshow(sttc_matrix, cmap='coolwarm', norm=log_norm)
            plt.colorbar(label='Log-STTC')
            plt.xlabel('Neuron')
            plt.ylabel('Neuron')
            plt.title(f"STTC Heatmap (Log Scale): {dataset_name} (Min={log_norm.vmin:.2e}, Max={log_norm.vmax:.2e})")
            plt.tight_layout()

            plot_name = os.path.join(output_path, f"sttc_log_{dataset_name}.png") #try to do this in the future, maybe easier to read
            plt.savefig(plot_name)
            plt.close()

    def plot_sttc_vmin_vmax(self, output_path, dataset_name):
        """
        Generate an STTC heatmap with dynamic vmin and vmax boundaries for all datasets.
        """
        print(f"Plotting STTC heatmap with dynamic vmin/vmax for {dataset_name}")
        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            sttc_matrix = self.compute_sttc_matrix(train, duration)
            vmin = np.percentile(sttc_matrix, 5)
            vmax = np.percentile(sttc_matrix, 95)

            plt.figure(figsize=(10, 8))
            plt.imshow(sttc_matrix, cmap='coolwarm', vmin=vmin, vmax=vmax)
            plt.colorbar(label='STTC')
            plt.xlabel('Neuron')
            plt.ylabel('Neuron')
            plt.title(f"STTC Heatmap (Percentile Range: {vmin:.2f}-{vmax:.2f}): {dataset_name}")
            plt.tight_layout()

            plot_name = os.path.join(output_path, f"sttc_vmin_vmax_{dataset_name}.png")
            plt.savefig(plot_name)
            plt.close()

    def plot_sttc_thresh(self, output_path, dataset_name):
        from matplotlib import cm
        """
        Generate an STTC heatmap with threshold-based shading for all datasets.
        """
        print(f"Plotting STTC heatmap with threshold-based shading for {dataset_name}")
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
            plt.title(f'STTC Heatmap (Thresholded: >{threshold_lower:.2f}): {dataset_name}')
            plt.tight_layout()

            plot_name = os.path.join(output_path, f"sttc_thresh_{dataset_name}.png")
            plt.savefig(plot_name)
            plt.close()

    def plot_kde_pdf(self, output_path, dataset_name):
        """
        Generate KDE and PDF plots for STTC values for all datasets.
        """
        print(f"Plotting STTC KDE and PDF for {dataset_name}")
        for train, duration in zip(self.trains, self.durations):
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
            plt.title(f'STTC KDE and PDF: {dataset_name}')
            plt.legend()
            plt.tight_layout()

            plot_name = os.path.join(output_path, f"kde_pdf_{dataset_name}.png")
            plt.savefig(plot_name)
            plt.close()

    def plot_comparison_kde_pdf(self, output_path, base_names):
        """
        Generate comparison KDE/PDF overlay plots for all datasets.
        """
        import scipy.stats
        import matplotlib.cm as cm
        print("Generating comparison overlay plot for STTC KDE and PDF")

        fig, ax_kde = plt.subplots(figsize=(12, 6))
        cmap = cm.get_cmap('tab10') #get colormap

        ax_pdf = ax_kde.twinx()  # secondary y-axis for PDF Density

        legend_items = []  # to store legend handles and labels in order

        for idx, (train, duration, name) in enumerate(zip(self.trains, self.durations, base_names)):
            sttc_matrix = self.compute_sttc_matrix(train, duration)
            sttc_values = self.get_upper_triangle_values(sttc_matrix)

            if sttc_values.size == 0:
                print(f"Warning: No valid STTC values for dataset {name_cleaned}. Skipping plot.")
                continue

            name_cleaned = name.replace('_acqm', '') #cleaning up the name to make legend smaller
            color = cmap(idx %10) # assign color based on index

            # KDE computation
            kde = scipy.stats.gaussian_kde(sttc_values)
            x_range = np.linspace(min(sttc_values), max(sttc_values), 500)
            kde_values = kde(x_range)

            # overlay KDE and PDF
            kde_line = ax_kde.plot(x_range, kde_values, label=f'{name_cleaned} (KDE)', alpha=0.7, color=color)
            pdf_bars= ax_pdf.hist(sttc_values, bins=50, density=True, alpha=0.3, label=f'{name_cleaned} (PDF)', color=color)

            # append to legend items in order
            legend_items.append((kde_line, f'{name} (KDE)'))
            legend_items.append((pdf_bars[2][0], f'{name} (PDF)'))  # use the first patch of the histogram

        # add threshold annotations
        line_thresh_02 = ax_kde.axvline(x=0.2, color='pink', linestyle='--', label='Threshold: |STTC|=0.2')
        line_thresh_05 = ax_kde.axvline(x=0.5, color='red', linestyle='--', label='Threshold: |STTC|=0.5')

        # append threshold lines to legend items
        legend_items.append((line_thresh_02, 'Threshold: |STTC|=0.2'))
        legend_items.append((line_thresh_05, 'Threshold: |STTC|=0.5'))

        # set axis labels and title
        ax_kde.set_xlabel('STTC Values')
        ax_kde.set_ylabel('KDE: Smoothed Density per Unit (Area=1)')
        ax_pdf.set_ylabel('PDF: Density per Bin (Area=1)')

        # set the custom legend
        handles, labels = zip(*legend_items)
        ax_kde.legend(handles, labels, loc='best')

        plt.title('Comparison: STTC KDE and PDF Across Datasets')
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
        print(f"Generating pairwise linear comparison for {metric}")

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
                    valid_intervals = [x for x in all_intervals if np.isfinite(1 / x)]
                    values.append(np.array(valid_intervals))
                return values
            elif metric == "firing_rate":
                return self.firing_rates_list
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        # compute metric values for all datasets
        metric_values_list = compute_metric(metric)

        # pairwise comparisons
        for (i, j) in combinations(range(len(base_names)), 2):
            
            #clean names
            name_1 = self.cleaned_names[i]
            name_2 = self.cleaned_names[j]

            dataset_1 = metric_values_list[i]
            dataset_2 = metric_values_list[j]

            min_size = min(len(dataset_1), len(dataset_2))
            dataset_1 = dataset_1[:min_size]
            dataset_2 = dataset_2[:min_size]

            # scatter plot
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

    def plot_global_metric_comparison(self, output_path, base_names, metric):
        """
        Global linear comparison of a given metric across all datasets.

        Args:
            output_path (str): Path to save the plots.
            base_names (list): Names of the datasets.
            metric (str): Metric to compute and compare ("sttc", "isi", "firing_rate").
        """
        # skip if fewer than three datasets 
        import os

        if len(base_names) < 3:
            print("Global comparison requires at least three datasets. Skipping.")
            return
        
        cleaned_names = self.cleaned_names

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
        for dataset, cleaned_name in zip(metric_values_list, cleaned_names):
            plt.scatter([cleaned_name] * len(dataset), dataset, label=cleaned_name, alpha=0.7)

        # add labels and save the plot
        plt.xlabel("Dataset")
        plt.ylabel(f"{metric.capitalize()} Values")
        plt.title(f"Global Comparison: Point Level Distribution of {metric.capitalize()} Across All Datasets")
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(output_path, f"{metric}_global_comparison.png")
        plt.savefig(plot_filename)
        plt.close()


    def calculate_static_fr_pca_whole(self, subset_neurons=None):
        """
        Calculate PCA for static firing rates across all datasets, truncating to the size of the smallest dataset.

        Parameters:
            subset_neurons (int, optional): Number of top neurons (by mean firing rate) to include.
                                            If None, all neurons are included.

        Returns:
            tuple: (pca_result, explained_variance)
                - pca_result: Transformed PCA coordinates (Neurons x PCs).
                - explained_variance: Variance explained by each PC.
        """

        from sklearn.decomposition import PCA

        print("Calculating PCA for static firing rates (whole dataset)")

        # construct firing rate matrices for all datasets
        firing_rate_matrices = [
            np.array(firing_rates).reshape(-1, 1) for firing_rates in self.firing_rates_list
        ]

        # determine the minimum number of neurons across datasets
        min_neurons = min(matrix.shape[0] for matrix in firing_rate_matrices)

        # truncate matrices to the size of the smallest dataset
        truncated_matrices = [
            matrix[:min_neurons, :] for matrix in firing_rate_matrices
        ]

        # combine truncated matrices into a single matrix (neurons x datasets)
        combined_matrix = np.hstack(truncated_matrices)

        # normalize by the total number of neurons
        total_neurons = combined_matrix.shape[0]
        combined_matrix = combined_matrix / total_neurons

        # optionally filter top neurons by mean firing rate
        if subset_neurons:
            mean_firing_rates = np.mean(combined_matrix, axis=1)
            top_indices = np.argsort(-mean_firing_rates)[:subset_neurons]
            combined_matrix = combined_matrix[top_indices, :]

        # check for valid firing rate matrix dimensions
        if combined_matrix.shape[0] < 2 or combined_matrix.shape[1] < 2:
            raise ValueError("Firing rate matrix is too small for PCA. Ensure sufficient neurons and datasets are included.")

        # PCA
        pca = PCA()
        pca_result = pca.fit_transform(combined_matrix)
        explained_variance = pca.explained_variance_ratio_

        return pca_result, explained_variance



    def calculate_individual_time_binned_pca(self, bin_size=0.1, subset_neurons=None):
        """
        Calculate PCA on time-binned firing rates for each dataset independently.

        Parameters:
            bin_size (float): Time bin size in seconds.
            subset_neurons (int, optional): Number of top neurons (by mean firing rate) to include.
                                            If None, all neurons are included.

        Returns:
            list of tuples: [(pca_result, explained_variance, time_bins, dataset_name), ...]
                Each tuple contains:
                    - pca_result: Transformed PCA coordinates (time bins x PCs).
                    - explained_variance: Variance explained by each PC.
                    - time_bins: Array of time bin centers.
                    - dataset_name: Name of the dataset.
        """
        from sklearn.decomposition import PCA
        print("Calculating PCA for individual time-binned firing rates")
        results = []

        for train, duration, dataset_name in zip(self.trains, self.durations, self.input_paths):
            # Determine bin edges based on duration
            num_bins = int(duration // bin_size) + 1
            bin_edges = np.linspace(0, duration, num_bins)

            # Compute time-binned firing rates
            binned_rates = np.array([
                np.histogram(neuron_spikes, bins=bin_edges)[0] / bin_size
                for neuron_spikes in train
            ]).T
            binned_rates /= len(train)  # Normalize by number of neurons in the dataset

            # Optionally filter top neurons by mean firing rate
            if subset_neurons:
                mean_firing_rates = np.mean(binned_rates, axis=0)
                top_indices = np.argsort(-mean_firing_rates)[:subset_neurons]
                binned_rates = binned_rates[:, top_indices]

            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(binned_rates)
            explained_variance = pca.explained_variance_ratio_

            # Store results
            time_bins = bin_edges[:-1] + (bin_size / 2)
            results.append((pca_result, explained_variance, time_bins, os.path.basename(dataset_name)))

        return results

    def calculate_static_pca_individual(self, subset_neurons=None):
        """
        Calculate PCA for static firing rates for each dataset independently.

        Parameters:
            subset_neurons (int, optional): Number of top neurons (by mean firing rate) to include.
                                            If None, all neurons are included.

        Returns:
            list: List of tuples (pca_result, explained_variance, dataset_name) for each dataset.
        """
        from sklearn.decomposition import PCA
        print("Calculating PCA for static firing rates (individual datasets)")
        results = []

        for firing_rates, dataset_name in zip(self.firing_rates_list, self.input_paths):
            firing_rate_matrix = np.array(firing_rates).reshape(-1, 1)  # Reshape for PCA

            # Optionally filter top neurons by mean firing rate
            if subset_neurons:
                top_indices = np.argsort(-firing_rate_matrix.flatten())[:subset_neurons]
                firing_rate_matrix = firing_rate_matrix[top_indices]

            # Normalize by the number of neurons
            firing_rate_matrix /= len(firing_rates)

            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(firing_rate_matrix)
            explained_variance = pca.explained_variance_ratio_

            results.append((pca_result, explained_variance, os.path.basename(dataset_name)))

        return results

    def plot_static_fr_pca_whole(self, pca_result, explained_variance, output_path):
        """
        Plot PCA results for static firing rates.

        Parameters:
            pca_result (numpy.ndarray): Transformed PCA coordinates (Neurons x PCs).
            explained_variance (numpy.ndarray): Variance explained by each PC.
            output_path (str): Directory to save the PCA plots.
        """
        print("Plotting PCA results for static firing rates (whole dataset)")
        # scatter Plot: neurons in PCA space (PC1 vs PC2)
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA of Firing Rate Variance (PC1 vs PC2, All Datasets) (Truncated to smallest dataset)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "pca_scatter_plot_static_whole.png"))
        plt.close()

        # scree plot: variance explained by each PC
        plt.figure(figsize=(10, 6))
        plt.bar([1, 2, 3], explained_variance[:3], alpha=0.7)  # only plot the first three PCs
        plt.xticks([1,2,3])
        plt.xlabel("Principal Component")
        plt.ylabel("Proportion of Variance Explained")
        plt.title("Scree Plot of PCA Components (All Datasets) (Truncated to smallest dataset)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "pca_scree_plot_static_whole.png"))
        plt.close()

    def plot_individual_time_binned_pca(self, pca_results, output_path):
        """
        Plot PCA results for each dataset independently.

        Parameters:
            pca_results: List of PCA results (output of `calculate_individual_time_binned_pca`).
            output_path: Directory to save the plots.
        """
        print("Plotting PCA results for individual time-binned firing rates")
        for pca_result, explained_variance, time_bins, dataset_name in pca_results:

            # Scatter plot: PC1 vs PC2 (colored by time)
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=time_bins, cmap='viridis', alpha=0.7)
            plt.colorbar(label='Time (s)')
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"PCA Scatter: {dataset_name} (PC1 vs PC2)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{dataset_name}_pca_scatter.png"))
            plt.close()

            # Component time series (first few PCs)
            subplots_dir = os.path.join(output_path, "pca_time_series_plots")
            os.makedirs(subplots_dir, exist_ok=True)
            segment_length = int(len(time_bins) / 5)  # Divide into 5 segments
            for segment_idx in range(5):
                start_idx = segment_idx * segment_length
                end_idx = start_idx + segment_length

                plt.figure(figsize=(12, 6))
                for pc_idx in range(3):  # Plot first 3 PCs
                    plt.plot(time_bins[start_idx:end_idx], pca_result[start_idx:end_idx, pc_idx], label=f"PC{pc_idx + 1}")

                plt.xlabel("Time (s)")
                plt.ylabel("Component Value")
                plt.title(f"PCA Components Over Time (Segment {segment_idx + 1}/5)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(subplots_dir, f"pca_time_series_segment_{segment_idx + 1}.png"))
                plt.close()

            # Scree plot: explained variance
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
            plt.xlabel("Principal Component")
            plt.ylabel("Proportion of Variance Explained")
            plt.title(f"Scree Plot: {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{dataset_name}_pca_scree_plot.png"))
            plt.close()

    def plot_static_pca_individual(self, pca_results, output_path):
        """
        Plot PCA results for static firing rates for each dataset.

        Parameters:
            pca_results: List of PCA results (output of `calculate_static_pca_individual`).
            output_path: Directory to save the plots.
        """
        print("Plotting PCA results for static firing rates (individual datasets)")        
        for (pca_result, explained_variance, dataset_name) in pca_results:

            """
            IGNORE THIS FOR NOW 

           
            File "/app/aws_npz_plot_gen.py", line 1943, in run_all_analyses
                self.plot_static_pca_individual(results, comparison_dir)  # individual static PCA
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            File "/app/aws_npz_plot_gen.py", line 1717, in plot_static_pca_individual
                        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
                                                      ~~~~~~~~~~^^^^^^
            IndexError: index 1 is out of bounds for axis 1 with size 1
            
            ***tested on same dataset no error thrown. not an important plot, will ignore for now

            # Scatter Plot: PC1 vs PC2
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"PCA of Firing Rate Variance (PC1 vs PC2): {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_dir, f"{dataset_name}_pca_scatter.png"))
            plt.close()
            """


            # Scree plot: variance explained by each PC
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
            plt.xlabel("Principal Component")
            plt.ylabel("Proportion of Variance Explained")
            plt.title(f"Scree Plot: {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{dataset_name}_pca_scree_plot.png"))
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

        print("Analyzing and comparing burst characteristics across datasets")

        # use the pre-existing comparison directory
        comparison_dir = os.path.join(output_path, "burst_characteristics")
        os.makedirs(comparison_dir, exist_ok=True)

        # clean dataset names
        cleaned_dataset_names = self.cleaned_names

        # initialize storage for comparisons across datasets
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

        # comparison plots
        # inter-burst interval (~ISI) comparison
        plt.figure(figsize=(12, 6))
        max_isi = 0  # track the maximum ISI for axis adjustment
        for i, burst_intervals in enumerate(all_burst_intervals):
            if len(burst_intervals) > 0:
                max_isi = max(max_isi, max(burst_intervals))  # update max ISI for axis scaling
                plt.hist(burst_intervals, bins=50, alpha=0.5, label=cleaned_dataset_names[i], density=True)
        plt.xlabel('Inter-Burst Interval (s)')
        plt.ylabel('Density')
        plt.title('Comparison: Inter-Burst Interval Across Datasets')
        plt.legend()
        plt.xlim(0, max_isi * 1.1)  # set x-axis limit based on data
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "comparison_inter_burst_interval.png"))
        plt.close()

        # burst frequency vs. duration comparison
        plt.figure(figsize=(12, 6))
        for i, (burst_frequencies, burst_durations) in enumerate(zip(all_burst_frequencies, all_burst_durations)):
            plt.scatter(burst_frequencies, burst_durations, alpha=0.6, label=cleaned_dataset_names[i])
        plt.xlabel('Burst Frequency (Hz)')
        plt.ylabel('Burst Duration (s)')
        plt.title('Comparison: Burst Frequency vs. Duration Across Datasets')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "comparison_burst_freq_duration.png"))
        plt.close()

        # neuron participation comparison
        plt.figure(figsize=(12, 6))
        for i, neuron_participation in enumerate(all_neuron_participation):
            plt.hist(neuron_participation, bins=50, alpha=0.5, label=cleaned_dataset_names[i], density=True)
        plt.xlabel('Neuron Participation Count')
        plt.ylabel('Density')
        plt.title('Comparison: Neuron Participation Across Datasets')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "comparison_neuron_participation.png"))
        plt.close()

    def plot_neuron_status_overlay(self, output_path, dataset_names):
        """
        Overlay footprint plots showing neuron presence, recovery, loss, and uniqueness across datasets.
        
        Parameters:
            output_path (str): Path to save the generated plot.
            dataset_names (list): Names of the datasets in chronological order.
        """
        from collections import defaultdict

        print("Generating neuron status overlay plot...")
        plt.close('all')

        # initialize neuron position tracking
        position_map = defaultdict(list)
        total_counts = []

        # build the position map
        for day_index, neuron_data in enumerate(self.neuron_data_list):
            for neuron_id, neuron_info in neuron_data.items():
                position = tuple(neuron_info['position'])
                position_map[position].append(day_index)
            total_counts.append(len(neuron_data))

        # categorize neurons
        all_days = set(range(len(self.neuron_data_list)))
        neuron_categories = {
            "Present in All": [],
            "Lost": [],
            "Recovered": [],
            "Unique": []
        }

        for position, days_present in position_map.items():
            days_present_set = set(days_present)
            if days_present_set == all_days:
                neuron_categories["Present in All"].append(position)
            elif days_present_set < all_days and max(days_present_set) < len(all_days) - 1:
                neuron_categories["Lost"].append(position)
            elif len(days_present_set) > 1 and min(days_present_set) > 0:
                neuron_categories["Recovered"].append(position)
            elif len(days_present_set) == 1:
                neuron_categories["Unique"].append(position)

        # plot neurons with color coding
        plt.figure(figsize=(12, 10))
        color_map = {
            "Present in All": "green",
            "Lost": "red",
            "Recovered": "blue",
            "Unique": "orange"
        }

        for category, positions in neuron_categories.items():
            positions = np.array(positions)
            if len(positions) > 0:
                plt.scatter(positions[:, 0], positions[:, 1], label=f"{category} ({len(positions)})",
                            color=color_map[category], alpha=0.6, edgecolors='k')

        # add title, labels, and legend
        plt.title("Neuron Status Overlay Across Datasets", fontsize=16)
        plt.xlabel(r"Horizontal Position ($\mu$m)", fontsize=14)
        plt.ylabel(r"Vertical Position ($\mu$m)", fontsize=14)
        plt.legend(title="Neuron Status", fontsize=12, title_fontsize=12)

        # add total neuron count for each day as text at the bottom
        text_lines = [f"Day {i + 1} ({name}): {count} neurons"
                    for i, (name, count) in enumerate(zip(dataset_names, total_counts))]
        plt.figtext(0.1, 0.01, "\n".join(text_lines), fontsize=10, ha="left", va="bottom", wrap=True)

        # save plot
        output_file = os.path.join(output_path, "neuron_status_overlay.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Neuron status overlay plot saved to {output_file}")


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
            self.footprint_overlay_fr(dataset_dir, dataset_name)
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
            self.sttc_violin_plot_by_firing_rate_individual(dataset_dir, dataset_name)
            self.sttc_violin_plot_by_proximity_individual(dataset_dir, dataset_name)

            if perform_pca:
                  #creating a pca subdirectory for each dataset
                pca_dir = os.path.join(dataset_dir, "pca")
                os.makedirs(pca_dir, exist_ok=True)

                # define a subset of neurons, if needed
                subset_neurons = None

                # individual PCA calculations
                pca_results = self.calculate_individual_time_binned_pca(bin_size=0.1, subset_neurons=subset_neurons)  # individual time-binned PCA
                results = self.calculate_static_pca_individual(subset_neurons=subset_neurons)  # individual static PCA
                # individual PCA plotting
                self.plot_individual_time_binned_pca(pca_results, pca_dir)  # individual time-binned PCA
                self.plot_static_pca_individual(results, pca_dir)  # individual static PCA



        # generate comparison plots
        self.sttc_violin_plot_by_firing_rate_compare(comparison_dir, base_names)
        self.sttc_violin_plot_by_proximity_compare(comparison_dir, base_names)
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
                self.plot_global_metric_comparison(global_dir, base_names, metric)


        # optional PCA analysis
        if perform_pca:
            # define the subset of neurons, if needed 
            subset_neurons = None  # replace with an integer to limit the number of neurons or leave as None
            # aggregate PCA calculations
            pca_result, explained_variance = self.calculate_static_fr_pca_whole(subset_neurons=subset_neurons)  # static aggregate PCA
            # aggregate PCA plotting
            self.plot_static_fr_pca_whole(pca_result, explained_variance, comparison_dir)  # averaged PCA

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
    analysis = SpikeDataAnalysis(input_s3)
    combined_name = "_".join(analysis.cleaned_names[:2])  # use original (cleaned) names for combined_name

    # create local temporary directories for processing
    local_output_folder = f'/tmp/output_plots_{combined_name}'


    zip_filename = analysis.run_all_analyses(local_output_folder, analysis.cleaned_names, perform_pca=perform_pca, cleanup=cleanup)

    output_s3 = os.path.join(output_s3, os.path.basename(zip_filename))
    # upload zip to S3
    print(f"Uploading {zip_filename} to {output_s3}")
    wr.upload(zip_filename, output_s3)

    print("Analysis complete. Results uploaded to S3.")




# main
if __name__ == '__main__':
    main()

