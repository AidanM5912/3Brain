import argparse
import braingeneers.utils.s3wrangler as wr
import collections
from datetime import datetime
import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
import scipy.stats
import scipy.ndimage
import sklearn.decomposition
import sklearn.cluster
import seaborn as sns
import sys
import tempfile
import uuid
import zipfile


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
        self.normalized_firing_rates_list = []
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
            self.normalized_firing_rates = self.normalized_firing_rates_list[0]
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

            # extract spike times and sampling rate
            spike_times = data["train"].item()
            sampling_rate = data["fs"]
            train = [times / sampling_rate for _, times in spike_times.items()]

            # store data and attributes
            self.data_list.append(data)
            self.trains.append(train)
            self.durations.append(max([max(times) for times in train]))
            firing_rates = [len(neuron_spikes) / max([max(times) for times in train]) for neuron_spikes in train]
            self.firing_rates_list.append(firing_rates)
            normalized_firing_rates = [rate / len(train) for rate in firing_rates]  # normalize by the total number of neurons
            self.normalized_firing_rates_list.append(normalized_firing_rates)
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

        # normalize to lowercase for case-insensitive comparison
        normalized_names = [re.sub(r'\W+', '_', name.lower()) for name in all_base_names]  # replace non-alphanumeric with "_"
        base_name_normalized = re.sub(r'\W+', '_', base_name.lower())

        # split names into components
        split_names = [set(name.split("_")) for name in normalized_names]
        common_parts = set.intersection(*split_names)  # find common components across all names

        # remove common parts and rejoin
        cleaned_parts = [part for part in base_name_normalized.split("_") if part not in common_parts]
        cleaned_name = "_".join(cleaned_parts)

        # return the original name if cleaning results in an empty string
        return cleaned_name if cleaned_name else base_name



    def raster_plot(self, output_path):
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting raster plot for {dataset_name}")

            fig, ax = plt.subplots(figsize=(10, 8))
            y = 0
            for vv in train:
                plt.scatter(vv, [y] * len(vv), marker="|", c='k', s=4, alpha=0.7)
                y += 1

            num_neurons = len(train)
            tick_spacing = 1 if num_neurons <= 50 else 2 if num_neurons <= 100 else 3 if num_neurons <= 150 else 5 if num_neurons <= 200 else 6 if num_neurons <= 250 else 7 if num_neurons <= 300 else 9 if num_neurons <= 400 else 10 if num_neurons <= 500 else 15
            ax.set_yticks(range(1, num_neurons + 1, tick_spacing))

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron Index')
            ax.set_xlim(0, self.durations[i])

            secax = ax.secondary_xaxis('top')
            secax.set_xlabel("Time (Hours)")
            xticks = ax.get_xticks()
            secax.set_xticks(xticks)
            secax.set_xticklabels([f"{x / 3600:.2f}" for x in xticks])

            ax.set_title(f"Raster Plot: {dataset_name}, Total units: {num_neurons}")
            plt.savefig(os.path.join(output_path, f"raster_{dataset_name}.png"))
            plt.close(fig)

    def footprint_overlay_fr(self, output_path):
        plt.close('all')
        for i, neuron_data in enumerate(self.neuron_data_list):

            dataset_name = self.cleaned_names[i]
            print(f"Plotting footprint overlay with firing rates for {dataset_name}")

            neuron_x, neuron_y, firing_rates = [], [], []

            for j, neuron in enumerate(neuron_data.values()):
                x, y = neuron['position']
                if (x, y) != (0, 0):  # exclude the calibration electrode
                    neuron_x.append(x)
                    neuron_y.append(y)
                    firing_rates.append(self.firing_rates_list[i][j])

            firing_rates = np.array(firing_rates)
            legend_rates = np.percentile(firing_rates, [50, 75, 90, 98])

            plt.figure(figsize=(11, 9))
            plt.scatter(neuron_x, neuron_y, s=firing_rates * 300, alpha=0.4, c='r', edgecolors='none')
        
            for rate in legend_rates:
                plt.scatter([], [], s=rate * 300, c='r', alpha=0.4, label=f'{rate:.2f} Hz')

            plt.legend(scatterpoints=1, frameon=True, labelspacing=1.4, handletextpad=0.8, borderpad=0.92, title='Firing Rate', loc = 'best', title_fontsize=10, fontsize=10)
            plt.xlabel(r"Horizontal Position ($\mu$m)")
            plt.ylabel(r"Vertical Position ($\mu$m)")
            plt.title(f"Footprint Plot with Firing Rates: {dataset_name}, Total units: {len(neuron_x)}")
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


    def plot_smoothed_population_fr(self, output_path):
        plt.close('all')
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting smoothed population firing rate{dataset_name}")

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

    def overlay_fr_raster(self, output_path):
        plt.close('all')
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
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
    def compute_sttc_matrix(spike_train, length, delt=.02):

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

    def sttc_plot(self, output_path):
        plt.close('all')
        for i, train in enumerate(self.trains):

            dataset_name = self.cleaned_names[i]
            print(f"Plotting STTC matrix for {dataset_name}")

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

    def sttc_violin_plot_by_firing_rate_individual(self, output_path):
        #generate STTC violin plots for an individual dataset by firing rate.
        
        plt.close('all')

        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            print(f"Generating STTC violin plots by firing rate for {dataset_name}")
            groups, low_threshold, high_threshold = self.group_neurons_by_firing_rate(self.normalized_firing_rates_list[i])
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

    def sttc_violin_plot_by_firing_rate_compare(self, output_path):
        #generate a combined STTC violin plot by firing rate across multiple datasets.
        print(f"Generating combined STTC violin plots by firing rate for all datasets")
        plt.close('all')

        combined_sttc_values = []
        combined_labels = []

        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            groups, low_threshold, high_threshold = self.group_neurons_by_firing_rate(self.normalized_firing_rates_list[i])

            for group_name, indices in groups.items():
                sttc_matrix = self.compute_sttc_matrix([train[j] for j in indices], self.durations[i])
                combined_sttc_values.append(self.get_upper_triangle_values(sttc_matrix))
                if group_name == 'low':
                    combined_labels.append(f"{dataset_name} - Low (<{low_threshold:.2f} Hz)")
                elif group_name == 'medium':
                    combined_labels.append(f"{dataset_name} - Medium ({low_threshold:.2f}-{high_threshold:.2f} Hz)")
                elif group_name == 'high':
                    combined_labels.append(f"{dataset_name} - High (≥{high_threshold:.2f} Hz)")

        # generate violin plot
        plt.figure(figsize=(12, 8))
        plt.violinplot(combined_sttc_values, showmeans=True)
        plt.xticks(ticks=np.arange(1, len(combined_labels) + 1), labels=combined_labels, rotation=45, ha='right')
        plt.xlabel('Firing Rate Group')
        plt.ylabel('STTC Values')
        plt.title('STTC Violin Plot by Firing Rate Group Across Recordings')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_firing_rate_combined.png"))
        plt.close()

    def sttc_violin_plot_by_proximity_individual(self, output_path, distance_threshold=100):
        """
        Generate STTC violin plots for an individual dataset by proximity.

        Parameters:
            output_path (str): Directory to save plots.
            dataset_name (str): Name of the dataset.
            distance_threshold (int): Threshold for proximity grouping.
        """
        plt.close('all')

        # find index of the dataset
        for i, train in enumerate(self.trains):
            
            dataset_name = self.cleaned_names[i]
            print(f"Generating STTC violin plots by proximity for {dataset_name}")

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

            # generate the violin plot
            plt.figure(figsize=(12, 8))
            plt.violinplot(sttc_values, showmeans=True)
            plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha='right')
            plt.xlabel('Proximity Group')
            plt.ylabel('STTC Values')
            plt.title(f'STTC Violin Plot by Spatial Proximity: {dataset_name} (Proximity ≤ {distance_threshold} μm)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"sttc_violin_plot_by_proximity_{dataset_name}.png"))
            plt.close()

    def sttc_violin_plot_by_proximity_compare(self, output_path, distance_threshold=100):
        """
        Generate a combined STTC violin plot by proximity across multiple datasets.

        Parameters:
            output_path (str): Directory to save plots.
            dataset_names (list): Names of datasets for comparison.
            distance_threshold (int): Threshold for proximity grouping.
        """
        print(f"Generating combined STTC violin plots by proximity for all datasets")
        plt.close('all')

        combined_sttc_values = []
        combined_labels = []

        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
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


    def sttc_violin_plot_across_recordings(self, output_path):
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
            labels.append(self.cleaned_names[i])

        plt.figure(figsize=(12, 8))
        plt.violinplot(sttc_values_list, showmeans=True)
        plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha ='right')
        plt.xlabel('Recordings')
        plt.ylabel('STTC Values')
        plt.title('Violin Plot of STTC Values Across Recordings')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"sttc_violin_plot_across_recordings.png"))
        plt.close()

    def plot_firing_rate_histogram(self, output_path):
        plt.close('all')
        for i, firing_rates in enumerate(self.normalized_firing_rates_list):

            dataset_name = self.cleaned_names[i]
            print(f"Plotting firing rate histogram for {dataset_name}")

            plt.figure(figsize=(12, 6))
            plt.hist(firing_rates, bins=50, color='green', alpha=0.7)
            plt.xlabel('Firing Rate (Hz)')
            plt.ylabel('Count')
            plt.title(f'Firing Rate Histogram: {dataset_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"firing_rate_histogram_{dataset_name}.png"))
            plt.close()

    def plot_firing_rate_cdf(self, output_path):
        plt.close('all')
        for i, firing_rates in enumerate(self.normalized_firing_rates_list):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting firing rate CDF for {dataset_name}")
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

    def plot_isi_histogram(self, output_path, bins=50, log_scale=False, xlim=None, kde=False):
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
            dataset_name = self.cleaned_names[i]
            print(f"Plotting ISI histogram for {dataset_name}")
            # collect all ISI values
            all_intervals = []
            for neuron_spikes in train:
                all_intervals.extend(np.diff(neuron_spikes))

            # filter ISI values based on xlim
            if xlim:
                all_intervals = [isi for isi in all_intervals if xlim[0] <= isi <= xlim[1]]

            # create histogram
            plt.figure(figsize=(12, 6))
            if log_scale:
                # use logarithmic bins if log_scale is True
                bins = np.logspace(np.log10(min(all_intervals) + 1e-6), np.log10(max(all_intervals)), bins)
                plt.xscale('log')
            else:
                # use linear bins (adjusted to xlim if specified)
                bins = np.linspace(xlim[0], xlim[1], bins) if xlim else bins

            plt.hist(all_intervals, bins=bins, color='red', alpha=0.7, label='Histogram')

            # add KDE if enabled
            if kde:
                from scipy.stats import gaussian_kde
                density = gaussian_kde(all_intervals)
                xs = np.linspace(min(all_intervals), max(all_intervals), 500)
                plt.plot(xs, density(xs), color='blue', label='KDE')

            # set x-axis limits
            if xlim:
                plt.xlim(xlim)  # explicitly set the x-axis range

            # labels and title
            plt.xlabel('Inter-Spike Interval (s)')
            plt.ylabel('Count')
            plt.title(f'ISI Histogram: {dataset_name}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"isi_histogram_{dataset_name}_log_{log_scale}.png"))
            plt.close()



    def plot_cv_of_isi(self, output_path, bins=50, xlim=None, kde=False):
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
            dataset_name = self.cleaned_names[i]
            print(f"Plotting CV of ISI histogram for {dataset_name}")
            # calculate CV values
            cv_values = []
            for neuron_spikes in train:
                intervals = np.diff(neuron_spikes)
                if len(intervals) > 1:
                    cv = np.std(intervals) / np.mean(intervals)  # CV: std/mean
                    cv_values.append(cv)

            # filter CV values based on xlim
            if xlim:
                cv_values = [cv for cv in cv_values if xlim[0] <= cv <= xlim[1]]

            # create histogram
            plt.figure(figsize=(12, 6))
            bins = np.linspace(xlim[0], xlim[1], bins) if xlim else bins  # asdjust bins to range
            plt.hist(cv_values, bins=bins, color='orange', alpha=0.7, label='Histogram')

            # add KDE if enabled
            if kde:
                from scipy.stats import gaussian_kde
                density = gaussian_kde(cv_values)
                xs = np.linspace(min(cv_values), max(cv_values), 500)
                plt.plot(xs, density(xs), color='blue', label='KDE')

            # set x-axis limits if specified
            if xlim:
                plt.xlim(xlim)

            # labels and title
            plt.xlabel('Coefficient of Variation (CV)')
            plt.ylabel('Count')
            plt.title(f'Coefficient of Variation of ISI: {dataset_name}')
            plt.tight_layout()

            # save the plot
            plt.savefig(os.path.join(output_path, f"cv_of_isi_{dataset_name}.png"))
            plt.close()

    def plot_raw_population_fr(self, output_path, bin_size=1.0):
        plt.close('all')
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting raw population firing rate for {dataset_name}")
            combined_train = np.hstack(train)
            max_time = np.max(combined_train)
            bins = np.arange(0, max_time + bin_size, bin_size)
            firing_rate, _ = np.histogram(combined_train, bins=bins)

             # normalize by the number of neurons
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

    def plot_synchrony_index_over_time(self, output_path, window_size=10.0):
        plt.close('all')
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting synchrony index over time for {dataset_name}")
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

    def plot_sttc_over_time(self, output_path, window_size=10.0):
        plt.close('all')
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting STTC over time for {dataset_name}")
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
    
    def plot_footprint_sttc(self, output_path):
        plt.close('all')
        for i, (train, neuron_data) in enumerate(zip(self.trains, self.neuron_data_list)):
            dataset_name = self.cleaned_names[i]
            print(f"Generating footprint plot with STTC for {dataset_name}")

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
            plt.scatter(neuron_x, neuron_y, s=sttc_marker_size * 50, alpha=0.4, c='r', edgecolors='none')
        
            for rate in legend_rates:
                plt.scatter([], [], s=rate * 50, c='r', alpha=0.4, label=f'STTC Sum {rate /100:.2f}')

            plt.legend(scatterpoints=1, frameon=True, labelspacing=1.4, handletextpad=0.8, borderpad=0.92, title='STTC Sum', loc = 'best', title_fontsize=10, fontsize=10, )
            plt.xlabel(r"Horizontal Position ($\mu$m)")
            plt.ylabel(r"Vertical Position ($\mu$m)")
            plt.title(f"Footprint Plot with STTC Sum : {dataset_name}")
            plt.savefig(os.path.join(output_path, f"footprint_plot_sttc_sum_{dataset_name}.png"))
            plt.close()

    def plot_comparison_inverse_isi(self, output_path):
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

            # remove zero intervals before calculating inverse ISI
            all_intervals = np.array(all_intervals)
            all_intervals = all_intervals[all_intervals > 0]  # exclude zero or negative intervals

            if len(all_intervals) > 0:
                inverse_isi = 1 / all_intervals
                inverse_isi = inverse_isi[np.isfinite(inverse_isi)]  # exclude any infinities
                
                # normalize by the number of neurons in the dataset
                num_neurons = len(train)
                if num_neurons > 0:
                    inverse_isi /= num_neurons  # normalize firing rate

                # apply clipping to remove extreme outliers
                inverse_isi = inverse_isi[inverse_isi < 1000]  # exclude values above 1000 Hz

                # use logarithmic bins if appropriate
                bins = np.logspace(np.log10(1), np.log10(max(inverse_isi)), 50)  # logarithmic bins
                plt.hist(inverse_isi, bins=bins, alpha=0.5, label=self.cleaned_names[i], density=True)
                plt.xscale('log')  # set x-axis to logarithmic scale

        plt.xlabel("Normalized Instantaneous Firing Rate (Hz/neuron)")
        plt.ylabel("Density")
        plt.title("Comparison: Population-Level Inverse ISI Across Datasets (Normalized)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "comparison_population_inverse_isi_normalized.png"))
        plt.close()


    def plot_comparison_regular_isi(self, output_path):
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
            plt.hist(all_intervals, bins=100, alpha=0.5, label=self.cleaned_names[i], density=True)

        plt.xlabel("Inter-Spike Interval (s)")
        plt.ylabel("Density")
        plt.title("Comparison: Regular ISI Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "comparison_regular_isi.png"))
        plt.close()

    def plot_population_inverse_isi(self, output_path):
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting population-level inverse ISI for {dataset_name}")
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

    def plot_high_activity_rasters(self, output_path):
        """
        Generate raster plots for high-activity periods with smaller time windows.

        **************************************************
        Disabled for now due to overcreating plots.
        Moving out of Docker V2.1 and into a rasters doc.
        From now on high_acitivity_rasters will be generated with a seperate job
        **************************************************

        Parameters:
            output_path (str): Directory to save the raster plots.
            dataset_name (str): Name of the dataset.

        Saves:
            Raster plots for specified time windows centered on high-activity periods.
        """
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            print(f"Generating high-activity rasters for {dataset_name}")
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


    def plot_comparison_firing_rate_histogram(self, output_path, bins=50):
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
        all_firing_rates = np.hstack(self.normalized_firing_rates_list)
        min_firing_rate = np.min(all_firing_rates)
        max_firing_rate = np.max(all_firing_rates)

        bin_edges = np.linspace(min_firing_rate, max_firing_rate, bins+1)  # 50 bins

        plt.figure(figsize=(12, 6))
        for i, firing_rates in enumerate(self.normalized_firing_rates_list):
            plt.hist(
                firing_rates,
                bins=bin_edges,
                alpha=0.5,
                label=self.cleaned_names[i],
                density=True
            )

        plt.xlabel("Firing Rate (Hz)")
        plt.ylabel("Density")
        plt.title("Comparison: Firing Rate Histograms Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "comparison_firing_rate_histogram.png"))
        plt.close()

    def plot_sttc_log(self, output_path):
        from matplotlib.colors import LogNorm
        """
        Generate an STTC heatmap on a logarithmic scale for all datasets.
        """
        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting STTC heatmap with log scale for {dataset_name}")
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

    def plot_sttc_vmin_vmax(self, output_path):
        """
        Generate an STTC heatmap with dynamic vmin and vmax boundaries for all datasets.
        """
        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting STTC heatmap with dynamic vmin/vmax for {dataset_name}")
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

    def plot_sttc_thresh(self, output_path):
        from matplotlib import cm
        """
        Generate an STTC heatmap with threshold-based shading for all datasets.
        """
        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting STTC heatmap with threshold-based shading for {dataset_name}")
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

    def plot_kde_pdf(self, output_path):
        """
        Generate KDE and PDF plots for STTC values for all datasets.
        """
        for i, (train, duration) in enumerate(zip(self.trains, self.durations)):
            dataset_name = self.cleaned_names[i]
            print(f"Generating KDE and PDF plots for STTC values for {dataset_name}")
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

    def plot_comparison_kde_pdf(self, output_path,):
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

        for idx, (train, duration) in enumerate(zip(self.trains, self.durations)):
            name = self.cleaned_names[idx]
            sttc_matrix = self.compute_sttc_matrix(train, duration)
            sttc_values = self.get_upper_triangle_values(sttc_matrix)

            if sttc_values.size == 0:
                print(f"Warning: No valid STTC values for dataset {name}. Skipping plot.")
                continue

            color = cmap(idx %10) # assign color based on index

            # KDE computation
            kde = scipy.stats.gaussian_kde(sttc_values)
            x_range = np.linspace(min(sttc_values), max(sttc_values), 500)
            kde_values = kde(x_range)

            # overlay KDE and PDF
            kde_line = ax_kde.plot(x_range, kde_values, label=f'{name} (KDE)', alpha=0.7, color=color)
            pdf_bars= ax_pdf.hist(sttc_values, bins=50, density=True, alpha=0.3, label=f'{name} (PDF)', color=color)

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

    def plot_pairwise_linear_comparison(self, output_path, metric):
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
        for (i, j) in combinations(range(len(self.cleaned_names)), 2):
            
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

    def plot_global_metric_comparison(self, output_path, metric):
        """
        Global linear comparison of a given metric across all datasets.

        Args:
            output_path (str): Path to save the plots.
            base_names (list): Names of the datasets.
            metric (str): Metric to compute and compare ("sttc", "isi", "firing_rate").
        """
        # skip if fewer than three datasets 
        import os

        if len(self.cleaned_names) < 3:
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

    def analyze_burst_characteristics(self, output_path):
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

    def plot_raw_footprint(self, output_path):
        plt.close('all')
        for i, neuron_data in enumerate(self.neuron_data_list):
            dataset_name = self.cleaned_names[i]
            print(f"Generating raw footprint plot for {dataset_name}")
            neuron_x, neuron_y = [], []

            for j, neuron in enumerate(neuron_data.values()):
                x, y = neuron['position']
                if (x, y) != (0, 0):  # exclude the calibration electrode
                    neuron_x.append(x)
                    neuron_y.append(y)

            plt.figure(figsize=(11, 9))
            plt.scatter(neuron_x, neuron_y, c='b')
        
            plt.xlabel(r"Horizontal Position ($\mu$m)")
            plt.ylabel(r"Vertical Position ($\mu$m)")
            plt.title(f"Footprint Plot with Firing Rates: {dataset_name}")
            plt.savefig(os.path.join(output_path, f"footprint_plot_fr_{dataset_name}.png"))
            plt.close()

    def compute_cv2(self):
        """
        Compute the CV2 score for each neuron in each dataset.
        Returns a list (one per dataset) of arrays with CV2 scores.
        """
        cv2_list = []
        for train in self.trains:
            dataset_cv2 = []
            for neuron_spikes in train:
                isis = np.diff(neuron_spikes)
                if len(isis) < 2:
                    dataset_cv2.append(np.nan)
                else:
                    # CV2: 2*|ΔISI| / (ISIₙ + ISIₙ₊₁)
                    cv2_vals = 2 * np.abs(np.diff(isis)) / (isis[:-1] + isis[1:])
                    dataset_cv2.append(np.nanmean(cv2_vals))
            cv2_list.append(np.array(dataset_cv2))
        return cv2_list

    def plot_cv2_violin(self, output_path):
        plt.close('all')
        for i, cv2_vals in enumerate(self.compute_cv2()):
            cv2_list = self.compute_cv2()
            plt.figure(figsize=(12, 8))
            plt.violinplot(cv2_list, showmeans=True, showmedians=True)
            plt.xticks(ticks=np.arange(1, len(self.cleaned_names)+1), labels=self.cleaned_names, rotation=45)
            plt.xlabel("Dataset")
            plt.ylabel("CV2")
            plt.title("CV2 Score Distribution Across Datasets")
            plt.tight_layout()
            save_path = os.path.join(output_path, "cv2_violin.png")
            plt.savefig(save_path)
            plt.close()

    def plot_fr_heatmap(self, output_path):
        """
        For each dataset, compute a 2D heat map of average firing rate by position.
        Uses the neuron positions (from self.neuron_data_list) and the average firing rates
        (from self.firing_rates_list) to compute a weighted 2D histogram, applies Gaussian smoothing,
        and plots the result.
        """
        from scipy.ndimage import gaussian_filter
        plt.close('all')

        # Loop over datasets
        for i, neuron_data in enumerate(self.neuron_data_list):
            dataset_name = self.cleaned_names[i]
            print(f"Plotting firing rate heat map for {dataset_name}")
            
            # Collect positions and firing rates for valid neurons (exclude calibration electrode)
            positions = []
            frates = []
            for j, neuron in enumerate(neuron_data.values()):
                x, y = neuron['position']
                if (x, y) != (0, 0):
                    positions.append([x, y])
                    frates.append(self.firing_rates_list[i][j])
            positions = np.array(positions)
            frates = np.array(frates)
            
            # Define grid limits from the positions (with a margin)
            margin = 50
            x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin
            y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin
            
            # Define grid resolution (number of bins)
            n_bins = 150  # adjust as needed
            x_edges = np.linspace(x_min, x_max, n_bins+1)
            y_edges = np.linspace(y_min, y_max, n_bins+1)
            
            # Compute a weighted 2D histogram: in each bin, average the firing rate of neurons that fall there.
            # First, compute sum of rates and counts per bin.
            rate_sum, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=[x_edges, y_edges], weights=frates)
            counts, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=[x_edges, y_edges])
            
            # Avoid divide-by-zero: where counts==0, set average to 0.
            avg_rate = np.divide(rate_sum, counts, out=np.zeros_like(rate_sum), where=counts!=0)
            
            # Apply Gaussian smoothing
            sigma = 3  # adjust sigma in bin units
            smoothed_rate = gaussian_filter(avg_rate, sigma=sigma)
            
            # Plot heatmap (with origin lower so that y increases upward)
            plt.figure(figsize=(10, 8))
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
            plt.imshow(smoothed_rate.T, cmap='hot', interpolation='nearest', origin='lower', extent=extent)
            plt.colorbar(label="Average Firing Rate (Hz)")
            plt.xlabel("Horizontal Position (µm)")
            plt.ylabel("Vertical Position (µm)")
            plt.title(f"Firing Rate Heat Map: {dataset_name}")
            plt.tight_layout()
            save_path = os.path.join(output_path, f"firing_rate_heatmap_{dataset_name}.png")
            plt.savefig(save_path)
            plt.close()

    def plot_avg_distance_histogram(self, output_path, bins_range=(0, 1000), n_bins=50):
        """
        For each dataset, compute the average Euclidean distance (per neuron) to all other neurons,
        and plot a histogram (line plot) with the number of neurons (y-axis) versus distance (x-axis).
        
        Parameters:
            output_path : Directory where the plot will be saved.
            bins_range  : Tuple (min, max) for the x-axis in µm. Default is (0, 1000).
            n_bins      : Number of bins to use.
        """
        for i, neuron_data in enumerate(self.neuron_data_list):
            dataset_name = self.cleaned_names[i]
            positions = []
            for j, neuron in enumerate(neuron_data.values()):
                pos = neuron['position']
                # Use np.allclose to compare arrays
                if not np.allclose(pos, [0, 0]):
                    positions.append(pos)
            positions = np.array(positions)
            if positions.size == 0:
                print(f"No valid positions for dataset {dataset_name}")
                continue

            # Compute pairwise distances and then the average distance per neuron.
            dist_matrix = np.linalg.norm(positions[:, None] - positions, axis=2)
            np.fill_diagonal(dist_matrix, np.nan)
            avg_distance = np.nanmean(dist_matrix, axis=1)

            # Define fixed bins over bins_range
            bin_edges = np.linspace(bins_range[0], bins_range[1], n_bins+1)
            counts, _ = np.histogram(avg_distance, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            plt.figure(figsize=(8, 6))
            plt.plot(bin_centers, counts, marker='o', linestyle='-', color='blue')
            plt.xlabel("Average Distance (µm)")
            plt.ylabel("Number of Neurons")
            plt.title(f"Neuron Count vs. Average Distance: {dataset_name}")
            plt.xlim(bins_range)
            plt.tight_layout()
            save_path = os.path.join(output_path, f"avg_distance_histogram_{dataset_name}.png")
            plt.savefig(save_path)
            plt.close()


    def plot_avg_distance_histogram_overlay(self, output_path=None, bins_range=(0, 1000), n_bins=10):
        """
        Overlay the histograms of average distances for all datasets on one plot.
        
        Parameters:
            output_path : (Optional) Directory where the plot will be saved. If None, the figure is not saved.
            bins_range  : Tuple (min, max) for the x-axis in µm.
            n_bins      : Number of bins to use.
            
        Returns:
            fig, ax : The matplotlib figure and axes objects.
        """

        fig, ax = plt.subplots(figsize=(10, 8))
        bin_edges = np.linspace(bins_range[0], bins_range[1], n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Choose a colormap with as many distinct colors as datasets.
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.neuron_data_list)))
        
        for i, neuron_data in enumerate(self.neuron_data_list):
            dataset_name = self.cleaned_names[i]
            positions = []
            for neuron in neuron_data.values():
                pos = neuron['position']
                if not np.allclose(pos, [0, 0]):
                    positions.append(pos)
            positions = np.array(positions)
            if positions.size == 0:
                print(f"No valid positions for dataset {dataset_name}")
                continue
            dist_matrix = np.linalg.norm(positions[:, None] - positions, axis=2)
            np.fill_diagonal(dist_matrix, np.nan)
            avg_distance = np.nanmean(dist_matrix, axis=1)
            counts, _ = np.histogram(avg_distance, bins=bin_edges)
            ax.plot(bin_centers, counts, marker='o', linestyle='-', color=colors[i], label=dataset_name)
        
        ax.set_xlabel("Average Distance (µm)")
        ax.set_ylabel("Number of Neurons")
        ax.set_title("Overlay of Neuron Count vs. Average Distance")
        ax.set_xlim(bins_range)
        ax.legend()
        fig.tight_layout()
        
        if output_path is not None:
            save_path = os.path.join(output_path, "avg_distance_histogram_overlay.png")
            fig.savefig(save_path)
        
        return fig, ax



    def plot_line_avg_distance_vs_sttc_with_counts(self, output_path=None, bins_range=(0, 1000), n_bins=20):
        """
        For each dataset, bin neurons by their average distance and compute for each bin:
        - the average STTC (calculated as the sum of STTC values with all other neurons divided by (N-1))
        - the number of neurons in that bin.
        Then, plot a line graph of average STTC versus average distance along with a bar plot of the neuron count
        (using a twin y-axis).
        
        Parameters:
            output_path : (Optional) Directory where the plots will be saved.
            bins_range  : Tuple (min, max) for the x-axis (µm). Default is (0, 1000).
            n_bins      : Number of bins to use.
            
        Returns:
            figures  : A list of matplotlib figure objects (one per dataset).
            axes_list: A list of tuples (ax1, ax2) where ax1 is the primary axis and ax2 is its twin.
        """
        figures = []
        axes_list = []
        
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            duration = self.durations[i]
            sttc_mat = self.compute_sttc_matrix(train, duration)
            # Compute per-neuron average STTC (exclude self by subtracting diagonal)
            sttc_sum = np.sum(sttc_mat, axis=1) - np.diag(sttc_mat)
            avg_sttc_all = sttc_sum / (len(train) - 1)
            
            # Get valid neuron positions.
            neuron_data = self.neuron_data_list[i]
            positions = []
            for neuron in neuron_data.values():
                pos = neuron['position']
                if not np.allclose(pos, [0, 0]):
                    positions.append(pos)
            positions = np.array(positions)
            if positions.size == 0:
                print(f"No valid positions for dataset {dataset_name}")
                continue

            # Compute average distance per neuron.
            dist_matrix = np.linalg.norm(positions[:, None] - positions, axis=2)
            np.fill_diagonal(dist_matrix, np.nan)
            avg_distance = np.nanmean(dist_matrix, axis=1)
            
            # Restrict avg_sttc to valid neurons (assumed to be in the same order as positions).
            avg_sttc = np.array(avg_sttc_all)[np.arange(len(positions))]
            
            # Bin the neurons by average distance.
            bin_edges = np.linspace(bins_range[0], bins_range[1], n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mean_sttc_bins = []
            counts_bins = []
            for b in range(n_bins):
                mask = (avg_distance >= bin_edges[b]) & (avg_distance < bin_edges[b+1])
                if np.any(mask):
                    mean_sttc_bins.append(np.nanmean(avg_sttc[mask]))
                    counts_bins.append(np.sum(mask))
                else:
                    mean_sttc_bins.append(np.nan)
                    counts_bins.append(0)
            
            # Create the plot with a twin y-axis.
            fig, ax1 = plt.subplots(figsize=(8, 6))
            color1 = 'tab:blue'
            ax1.plot(bin_centers, mean_sttc_bins, marker='o', linestyle='-', color=color1, label="Avg STTC")
            ax1.set_xlabel("Average Distance (µm)")
            ax1.set_ylabel("Average STTC", color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.set_xlim(bins_range)
            ax1.set_title(f"Avg STTC vs. Average Distance with Unit Counts: {dataset_name}")
            
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.bar(bin_centers, counts_bins, width=(bin_edges[1] - bin_edges[0]) * 0.8,
                    alpha=0.3, color=color2, label="Neuron Count")
            ax2.set_ylabel("Number of Neurons", color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            fig.tight_layout()
            
            if output_path is not None:
                save_path = os.path.join(output_path, f"line_avg_distance_vs_sttc_with_counts_{n_bins}_{dataset_name}.png")
                fig.savefig(save_path)
            
            figures.append(fig)
            axes_list.append((ax1, ax2))
        
        return figures, axes_list

    def plot_line_avg_distance_vs_sttc(self, output_path=None):
        """
        For each dataset, compute for each valid neuron its average Euclidean distance to all other neurons
        and its average STTC (sum of pairwise STTC divided by [N-1]). Then, sort the neurons by average distance
        and plot a line graph of average STTC versus average distance.
        
        Parameters:
            output_path : (Optional) Directory where the plots will be saved.
            
        Returns:
            figures  : A list of matplotlib figure objects (one per dataset).
            axes_list: A list of axes objects corresponding to each figure.
        """

        figures = []
        axes_list = []
        
        for i, train in enumerate(self.trains):
            dataset_name = self.cleaned_names[i]
            duration = self.durations[i]
            # Compute the STTC matrix and then average STTC per neuron.
            sttc_mat = self.compute_sttc_matrix(train, duration)
            sttc_sum = np.sum(sttc_mat, axis=1) - np.diag(sttc_mat)
            avg_sttc_all = sttc_sum / (len(train) - 1)
            
            # Extract positions and record the corresponding indices.
            neuron_data = self.neuron_data_list[i]
            positions = []
            valid_indices = []
            for j, neuron in enumerate(neuron_data.values()):
                pos = neuron['position']
                if not np.allclose(pos, [0, 0]):
                    positions.append(pos)
                    valid_indices.append(j)
            positions = np.array(positions)
            if positions.size == 0:
                print(f"No valid positions for dataset {dataset_name}")
                continue

            # Compute pairwise distances among valid neurons and then the average distance per valid neuron.
            dist_matrix = np.linalg.norm(positions[:, None] - positions, axis=2)
            np.fill_diagonal(dist_matrix, np.nan)
            avg_distance = np.nanmean(dist_matrix, axis=1)
            
            # Restrict avg_sttc to valid neurons:
            avg_sttc = np.array(avg_sttc_all)[valid_indices]
            
            # Sort the neurons by their average distance.
            sort_indices = np.argsort(avg_distance)
            sorted_avg_distance = avg_distance[sort_indices]
            sorted_avg_sttc = avg_sttc[sort_indices]
            
            # Plot the line graph.
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(sorted_avg_distance, sorted_avg_sttc, marker='o', linestyle='-', color='darkgreen')
            ax.set_xlabel("Average Distance to Other Neurons (µm)")
            ax.set_ylabel("Average STTC")
            ax.set_title(f"Average STTC vs. Average Distance: {dataset_name}")
            fig.tight_layout()
            
            if output_path is not None:
                save_path = os.path.join(output_path, f"line_avg_distance_vs_sttc_{dataset_name}.png")
                fig.savefig(save_path)
            
            figures.append(fig)
            axes_list.append(ax)
        
        return figures, axes_list


    def randomize_raster(self, spike_trains, seed=None):
        """
        Randomizes spike times while preserving the total number of spikes per neuron. Does not perserve the isi
        
        Parameters:
        spike_trains : list of arrays
            Each entry is an array of spike times for a neuron.
        seed : int (optional)
            Random seed for reproducibility.

        Returns:
        randomized_trains : list of arrays
            Randomized spike trains with preserved firing rates.
        """
        rng = np.random.default_rng(seed)  # Random generator
        num_neurons = len(spike_trains)
        duration = max([max(times) for times in spike_trains if len(times) > 0])  # Get max time
        
        # Convert spike times to binary raster (bins=1 ms for simplicity)
        bin_size = 1  # 1ms bins
        num_bins = int(np.ceil(duration / bin_size))
        raster = np.zeros((num_neurons, num_bins), dtype=int)

        for i, spikes in enumerate(spike_trains):
            if len(spikes) > 0:
                indices = (spikes / bin_size).astype(int)
                raster[i, indices] = 1  # Assign spikes to bins

        # Shuffle spike times while preserving neuron firing rates
        randomized_raster = np.zeros_like(raster)
        for i in range(num_neurons):
            spike_count = np.sum(raster[i])  # Total spikes for this neuron
            spike_positions = rng.choice(num_bins, size=spike_count, replace=False)  # Shuffle spike locations
            randomized_raster[i, spike_positions] = 1  # Assign new spike locations

        # Convert back to spike time format
        randomized_trains = [np.where(randomized_raster[i])[0] * bin_size for i in range(num_neurons)]
        
        return randomized_trains
    
    def compute_filtered_sttc(self, spike_train, length, delt=20, n_shuffles=5, threshold_type="hard"):
        """
        Compute STTC for real and shuffled spike trains, filtering out non-significant values.

        Parameters:
        spike_train : list of arrays
            Spike times for each neuron.
        length : float
            Duration of recording.
        delt : int, optional
            STTC window parameter.
        n_shuffles : int, optional
            Number of times to shuffle the spike trains.
        threshold_type : str, optional
            "hard" - Keep STTC values greater than the shuffled mean.
            "statistical" - Keep STTC values greater than the 95th percentile of shuffled STTC.

        Returns:
        sttc_matrix_filtered : np.ndarray
            STTC matrix with only significant values kept.
        """
        real_sttc = self.compute_sttc_matrix(spike_train, length, delt)  # Compute real STTC

        shuffled_sttcs = []
        for _ in range(n_shuffles):
            randomized_train = self.randomize_raster(spike_train)
            shuffled_sttcs.append(self.compute_sttc_matrix(randomized_train, length, delt))
        
        shuffled_sttc_mean = np.mean(shuffled_sttcs, axis=0)
        shuffled_sttc_percentile = np.percentile(shuffled_sttcs, 95, axis=0)

        print("Sample real STTC values before filtering:")
        print(real_sttc[:5, :5])  # Print a small portion of the STTC matrix

        print("Sample shuffled STTC values before filtering:")
        print(shuffled_sttc_mean[:5, :5])  # Print the shuffled mean STTC

        print("Sample shuffled 95th Percentile STTC values:")
        print(shuffled_sttc_percentile[:5, :5])  # Print the shuffled 95th percentile

        #filter
        if threshold_type == "hard":
            sttc_pure = real_sttc - shuffled_sttc_mean  # Subtract shuffled STTC mean
            sttc_matrix_filtered = np.where(sttc_pure > 0, sttc_pure, 0)  # Keep only positive values
        elif threshold_type == "statistical":
            sttc_pure = real_sttc - shuffled_sttc_percentile  # Subtract 95th percentile shuffled STTC
            sttc_matrix_filtered = np.where(sttc_pure > 0, sttc_pure, 0)  # Keep only positive values
        else:
            raise ValueError("Invalid threshold_type. Choose 'hard' or 'statistical'.")
        
        print(f"Applying threshold: {threshold_type}")

        print(f"Total STTC values before filtering (non-NaN): {np.sum(~np.isnan(real_sttc))}")
        print(f"Total STTC values after filtering (non-NaN): {np.sum(~np.isnan(sttc_matrix_filtered))}")


        print(f"Real STTC mean: {np.nanmean(real_sttc):.4f}, min: {np.nanmin(real_sttc):.4f}, max: {np.nanmax(real_sttc):.4f}")
        print(f"Shuffled STTC mean: {np.nanmean(shuffled_sttc_mean):.4f}, min: {np.nanmin(shuffled_sttc_mean):.4f}, max: {np.nanmax(shuffled_sttc_mean):.4f}")
        print(f"95th Percentile of Shuffled STTC: {np.nanpercentile(shuffled_sttcs, 95):.4f}")

        return sttc_matrix_filtered
    

    def plot_sttc_vs_distance(self, output_path, n_bins=5, global_bins=None, filter_shuffled=True, threshold_type="hard"):
        """
        For each dataset, compute each neuron's average STTC (only retaining a connection if the real STTC 
        exceeds the average shuffled STTC value). Then, for neurons with valid positions (non-[0,0]),
        compute the average Euclidean distance to all other valid neurons. Using a fixed global range 
        (e.g. 200–1000 µm) divided into 5 bins, bin the neurons by distance and plot:
        - A violin plot (one violin per bin) of the neuron's average STTC.
        - A scatter plot of average STTC vs. average distance.

        Parameters:
        output_path    : str
            Directory where plots are saved.
        n_bins         : int, optional
            Number of distance bins (default 5).
        global_bins    : np.ndarray or list, optional
            Fixed bin edges (e.g. [200,360,520,680,840,1000]). If None, they are computed from the data.
        filter_shuffled: bool, optional
            Whether to filter STTC by comparing to the shuffled average.
        threshold_type : str, optional
            "hard" (keep connection if real STTC > average shuffled STTC) 
            or "statistical" (keep connection if real STTC > 95th percentile of shuffled STTC).
        """

        plt.close('all')

        for i, train in enumerate(self.trains):
            duration = self.durations[i]

            # 1. Compute the STTC matrix (filtered if requested)
            if filter_shuffled:
                sttc_mat = self.compute_filtered_sttc(train, duration, n_shuffles=5, threshold_type=threshold_type)
            else:
                sttc_mat = self.compute_sttc_matrix(train, duration)
            print(f"Dataset {self.cleaned_names[i]}: STTC matrix shape: {sttc_mat.shape}, non-NaN entries: {np.sum(~np.isnan(sttc_mat))}")

            # 2. For each neuron, compute its average STTC (excluding self).
            n_neurons = sttc_mat.shape[0]
            avg_sttc = np.empty(n_neurons)
            for j in range(n_neurons):
                # Remove the self-correlation (diagonal)
                row = np.delete(sttc_mat[j, :], j)
                avg_sttc[j] = np.nanmean(row)
            print(f"Computed average STTC across all neurons, overall mean: {np.nanmean(avg_sttc):.4f}")

            # 3. Extract valid neuron positions (drop only neurons at [0,0])
            neuron_data = self.neuron_data_list[i]
            positions_list = []
            valid_indices = []
            for j, neuron in enumerate(neuron_data.values()):
                pos = neuron['position']
                if not np.allclose(pos, [0, 0]):
                    positions_list.append(pos)
                    valid_indices.append(j)
            positions = np.array(positions_list)
            if positions.size == 0:
                print(f"No valid positions for dataset {self.cleaned_names[i]}")
                return

            # Restrict average STTC to valid neurons.
            avg_sttc_valid = avg_sttc[valid_indices]
            print(f"Valid neurons (with positions) count: {len(valid_indices)}; non-NaN avg STTC among them: {np.sum(~np.isnan(avg_sttc_valid))}")

            # 4. Compute each valid neuron's average distance to all other valid neurons.
            #    (This is done from the positions of valid neurons.)
            dist_matrix = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
            np.fill_diagonal(dist_matrix, np.nan)
            avg_distance = np.nanmean(dist_matrix, axis=1)
            print(f"Average distance: min={np.nanmin(avg_distance):.2f}, max={np.nanmax(avg_distance):.2f}")
            print("First 10 avg_distance values:", avg_distance[:10])

            # 5. Set up fixed global bins.
            if global_bins is None:
                # Compute from the data range if not provided
                global_bins = np.linspace(np.floor(np.nanmin(avg_distance)),
                                            np.ceil(np.nanmax(avg_distance)), n_bins + 1)
            else:
                global_bins = np.array(global_bins)
            print(f"Global bins used: {global_bins}")

            # 6. Bin neurons by avg_distance.
            bin_labels = [f"{global_bins[b]:.1f}-{global_bins[b+1]:.1f}" for b in range(n_bins)]
            sttc_bins = {label: [] for label in bin_labels}

            for b in range(n_bins):
                if b < n_bins - 1:
                    mask = (avg_distance >= global_bins[b]) & (avg_distance < global_bins[b+1])
                else:
                    mask = (avg_distance >= global_bins[b]) & (avg_distance <= global_bins[b+1])
                n_in_bin = np.sum(mask)
                print(f"Bin {b} ({bin_labels[b]}) - Neurons in bin: {n_in_bin}")
                if n_in_bin > 0:
                    sttc_bins[bin_labels[b]] = avg_sttc_valid[mask]
                else:
                    # Even if no neuron falls in this bin, assign an array with a single NaN,
                    # so that the bin will still appear (and colors/order remains constant).
                    sttc_bins[bin_labels[b]] = np.array([np.nan])

            # 7. Create a DataFrame for the violin plot.
            #    We want one row per neuron (with its bin label and its average STTC)
            bin_col = []
            sttc_col = []
            for label, vals in sttc_bins.items():
                bin_col.extend([label] * len(vals))
                sttc_col.extend(vals)
            df = pd.DataFrame({
                "Distance Bin": bin_col,
                "Average STTC": sttc_col
            })
            print(f"Total neurons for plotting (should equal valid neuron count): {len(df)}")

            ## 8. Violin Plot (binned)
            plt.figure(figsize=(8, 6))
            sns.violinplot(x="Distance Bin", y="Average STTC", data=df, inner="quartile", palette="Set3")
            plt.xlabel("Average Separation Distance (µm)")
            plt.ylabel("Average STTC")
            plt.ylim(0,1)
            plt.title(f"STTC by Distance: {self.cleaned_names[i]}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_path = os.path.join(output_path, f"sttc_vs_distance_violin_{self.cleaned_names[i]}.png")
            plt.savefig(save_path)
            plt.close()

            ## 9. Scatter Plot (unbinned)
            plt.figure(figsize=(8, 6))
            plt.scatter(avg_distance, avg_sttc_valid, c='darkorange', edgecolor='k', s=60, alpha=0.8)
            plt.xlabel("Average Distance (µm)")
            plt.ylabel("Average STTC")
            plt.ylim(0,1)
            plt.title(f"Average STTC vs. Average Distance: {self.cleaned_names[i]}")
            plt.tight_layout()
            save_path = os.path.join(output_path, f"sttc_vs_distance_{self.cleaned_names[i]}.png")
            plt.savefig(save_path)
            plt.close()


    def plot_sttc_heatmap(self, output_path, n_bins_distance=50, global_bins=None, n_bins_sttc=50, 
                        filter_shuffled=True, threshold_type="hard", heatmap_vmax=10):
        """
        Generates a heatmap of neuron pair counts over STTC and distance bins.
        
        For each dataset:
        - The filtered STTC matrix is computed (real minus shuffled, with negatives set to 0).
        - Valid neurons (positions ≠ [0,0]) are identified.
        - The pairwise distance matrix is computed from their positions.
        - Only the upper triangle (unique pairs) is used.
        - A 2D histogram is computed using fixed bins:
                * Distance: global_bins (e.g. computed from 200 to 1000 µm)
                * STTC: bins from 0 to 1.
        - The resulting count matrix is smoothed for better visual appearance.
        
        Parameters:
            output_path    : str
                Directory where plots are saved.
            n_bins_distance: int, optional
                Number of distance bins (if global_bins not provided).
            global_bins    : np.ndarray or list, optional
                Fixed bin edges for distance (e.g. [200,360,...,1000]). If None, computed from data.
            n_bins_sttc    : int, optional
                Number of STTC bins between 0 and 1 (default 50).
            filter_shuffled: bool, optional
                Whether to filter STTC using shuffled data.
            threshold_type : str, optional
                "hard" (real STTC > mean shuffled) or "statistical" (real STTC > 95th percentile of shuffled).
            heatmap_vmax   : float, optional
                Maximum color intensity for heatmap standardization.
        """
        from scipy.ndimage import gaussian_filter

        plt.close('all')

        for i, train in enumerate(self.trains):
            duration = self.durations[i]

            # Compute the filtered STTC matrix.
            if filter_shuffled:
                sttc_mat = self.compute_filtered_sttc(train, duration, n_shuffles=5, threshold_type=threshold_type)
            else:
                sttc_mat = self.compute_sttc_matrix(train, duration)
            print(f"Dataset {self.cleaned_names[i]}: STTC matrix shape: {sttc_mat.shape}, non-NaN: {np.sum(~np.isnan(sttc_mat))}")

            # Extract valid neurons and restrict matrices.
            neuron_data = self.neuron_data_list[i]
            positions_list = []
            valid_indices = []
            for j, neuron in enumerate(neuron_data.values()):
                pos = neuron['position']
                if not np.allclose(pos, [0, 0]):
                    positions_list.append(pos)
                    valid_indices.append(j)
            positions = np.array(positions_list)
            if positions.size == 0:
                print(f"No valid positions for dataset {self.cleaned_names[i]}")
                continue

            sttc_filtered = sttc_mat[np.ix_(valid_indices, valid_indices)]
            dist_matrix = np.linalg.norm(positions[:, None] - positions, axis=2)

            # Use only the upper triangle (unique pairs)
            triu_indices = np.triu_indices(len(valid_indices), k=1)
            sttc_values = sttc_filtered[triu_indices]
            distance_values = dist_matrix[triu_indices]

            # Define distance bins.
            if global_bins is None:
                global_bins = np.linspace(np.floor(np.nanmin(distance_values)),
                                        np.ceil(np.nanmax(distance_values)),
                                        n_bins_distance + 1)
            else:
                global_bins = np.array(global_bins)
            # Define STTC bins from 0 to 1.
            sttc_bins = np.linspace(0, 1, n_bins_sttc + 1)
            print("Global distance bins:", global_bins)
            print("STTC bins:", sttc_bins)

            # Compute a 2D histogram.
            heatmap_matrix, _, _ = np.histogram2d(distance_values, sttc_values, bins=[global_bins, sttc_bins])
            print("Raw heatmap matrix (counts):")
            print(heatmap_matrix)

            # Apply Gaussian smoothing.
            smoothed_matrix = gaussian_filter(heatmap_matrix, sigma=1.5)

            # Create tick labels, but only show a subset to avoid crowding.
            full_xticks = [f"{edge:.1f}" for edge in sttc_bins[:-1]]
            full_yticks = [f"{global_bins[k]:.1f}-{global_bins[k+1]:.1f}" for k in range(len(global_bins)-1)]
            # For instance, show every 10th label:
            xtick_labels = [full_xticks[i] if i % 10 == 0 else "" for i in range(len(full_xticks))]
            ytick_labels = [full_yticks[i] if i % 10 == 0 else "" for i in range(len(full_yticks))]

            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(smoothed_matrix, cmap="viridis", vmin=0, vmax=heatmap_vmax, fmt=".0f",
                            xticklabels=xtick_labels,
                            yticklabels=ytick_labels)
            ax.invert_yaxis()
            plt.xlabel("STTC Value")
            plt.ylabel("Distance (µm)")
            plt.title(f"Neuron Pair Counts: STTC vs Distance - {self.cleaned_names[i]}")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            save_path = os.path.join(output_path, f"sttc_heatmap_{self.cleaned_names[i]}.png")
            plt.savefig(save_path)
            plt.close()


    def run_all_analyses(self, output_folder, base_names, cleanup=True):
        """
        Execute all analyses for individual datasets and comparisons.
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
        for i, dataset_dir in enumerate(dataset_directories):
            
            #create rasters 
            #self.raster_plot(rasters_dir, dataset_name)
            #self.plot_high_activity_rasters(rasters_dir, dataset_name)

            #regular analysis
            self.raster_plot(dataset_dir)
            self.footprint_overlay_fr(dataset_dir)
            self.overlay_fr_raster(dataset_dir)
            self.plot_smoothed_population_fr(dataset_dir)
            self.sttc_plot(dataset_dir)
            self.plot_firing_rate_histogram(dataset_dir)
            self.plot_firing_rate_cdf(dataset_dir)
            self.plot_isi_histogram(dataset_dir)
            self.plot_cv_of_isi(dataset_dir)
            self.plot_raw_population_fr(dataset_dir)
            self.plot_synchrony_index_over_time(dataset_dir)
            self.plot_sttc_over_time(dataset_dir)
            self.plot_footprint_sttc(dataset_dir)
            self.plot_sttc_log(dataset_dir)
            self.plot_sttc_vmin_vmax(dataset_dir)
            self.plot_sttc_thresh(dataset_dir)
            self.plot_kde_pdf(dataset_dir)
            self.sttc_violin_plot_by_firing_rate_individual(dataset_dir)
            self.sttc_violin_plot_by_proximity_individual(dataset_dir)
            self.plot_raw_footprint(dataset_dir)

        # generate comparison plots
        self.sttc_violin_plot_by_firing_rate_compare(comparison_dir)
        self.sttc_violin_plot_by_proximity_compare(comparison_dir)
        self.sttc_violin_plot_across_recordings(comparison_dir)
        self.plot_comparison_inverse_isi(comparison_dir)
        self.plot_comparison_regular_isi(comparison_dir)
        self.plot_comparison_firing_rate_histogram(comparison_dir)
        self.plot_comparison_kde_pdf(comparison_dir)
        self.analyze_burst_characteristics(comparison_dir)

        #linear comparisons
        metrics = ["sttc", "firing_rate", "isi"]
        for metric in metrics:
            self.plot_pairwise_linear_comparison(pairwise_dir, metric)

        #global linear comparison (for 3+ datasets)
        if len(self.trains) >= 3:
            for metric in metrics:
                self.plot_global_metric_comparison(global_dir, metric)
                
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
    #optional flag for cleanup
    parser.add_argument("--cleanup", action="store_true", help="Delete the output folder after zipping")
    args = parser.parse_args()

    #access parsed arguments
    input_s3 = args.input_s3
    output_s3 = args.output_path
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


    zip_filename = analysis.run_all_analyses(local_output_folder, analysis.cleaned_names, cleanup=cleanup)

    output_s3 = os.path.join(output_s3, os.path.basename(zip_filename))
    # upload zip to S3
    print(f"Uploading {zip_filename} to {output_s3}")
    wr.upload(zip_filename, output_s3)

    print("Analysis complete. Results uploaded to S3.")




# main
if __name__ == '__main__':
    main()

