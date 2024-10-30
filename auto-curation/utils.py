import numpy as np
import io
import zipfile
import pandas as pd


def read_phy_files(path: str, fs=20000.0):
    """
    :param path: a s3 or local path to a zip of phy files.
    :return: SpikeData class with a list of spike time lists and neuron_data.
            neuron_data = {0: neuron_dict, 1: config_dict}
            neuron_dict = {"new_cluster_id": {"channel": c, "position": (x, y),
                            "amplitudes": [a0, a1, an], "template": [t0, t1, tn],
                            "neighbor_channels": [c0, c1, cn],
                            "neighbor_positions": [(x0, y0), (x1, y1), (xn,yn)],
                            "neighbor_templates": [[t00, t01, t0n], [tn0, tn1, tnn]}}
            config_dict = {chn: pos}
    """
    assert path[-3:] == 'zip', 'Only zip files supported!'
    import braingeneers.utils.smart_open_braingeneers as smart_open
    with smart_open.open(path, 'rb') as f0:
        f = io.BytesIO(f0.read())

        with zipfile.ZipFile(f, 'r') as f_zip:
            assert 'params.py' in f_zip.namelist(), "Wrong spike sorting output."
            with io.TextIOWrapper(f_zip.open('params.py'), encoding='utf-8') as params:
                for line in params:
                    if "sample_rate" in line:
                        fs = float(line.split()[-1])
            clusters = np.load(f_zip.open('spike_clusters.npy')).squeeze()
            templates_w = np.load(f_zip.open('templates.npy'))  # (cluster_id, samples, channel_id)
            wmi = np.load(f_zip.open('whitening_mat_inv.npy'))
            channels = np.load(f_zip.open('channel_map.npy')).squeeze()
            spike_templates = np.load(f_zip.open('spike_templates.npy')).squeeze()
            spike_times = np.load(f_zip.open('spike_times.npy')).squeeze() / fs * 1e3  # in ms
            positions = np.load(f_zip.open('channel_positions.npy'))
            amplitudes = np.load(f_zip.open("amplitudes.npy")).squeeze()
            if 'cluster_info.tsv' in f_zip.namelist():
                cluster_info = pd.read_csv(f_zip.open('cluster_info.tsv'), sep='\t')
                cluster_id = np.array(cluster_info['cluster_id'])
                # select clusters using curation label, remove units labeled as "noise"
                # find the best channel by amplitude
                labeled_clusters = cluster_id[cluster_info['group'] != "noise"]
            else:
                labeled_clusters = np.unique(clusters)

    # unwhite the templates before finding the best channel!
    templates = np.dot(templates_w, wmi)

    df = pd.DataFrame({"clusters": clusters, "spikeTimes": spike_times, "amplitudes": amplitudes})
    cluster_agg = df.groupby("clusters").agg({"spikeTimes": lambda x: list(x),
                                              "amplitudes": lambda x: list(x)})
    cluster_agg = cluster_agg[cluster_agg.index.isin(labeled_clusters)]

    cls_temp = dict(zip(clusters, spike_templates))
    neuron_dict = dict.fromkeys(np.arange(len(labeled_clusters)), None)

    for i in range(len(labeled_clusters)):
        c = labeled_clusters[i]
        temp = templates[cls_temp[c]]
        amp = np.max(temp, axis=0) - np.min(temp, axis=0)
        sorted_idx = np.argsort(amp)[::-1]
        nbgh_chan_idx = sorted_idx[:12]
        nbgh_temps = temp.transpose()[sorted_idx]
        best_chan_temp = nbgh_temps[0]
        nbgh_channels = channels[nbgh_chan_idx]
        nbgh_postions = [tuple(positions[idx]) for idx in nbgh_chan_idx]
        best_channel = nbgh_channels[0]
        best_position = nbgh_postions[0]
        cls_amp = cluster_agg["amplitudes"][c]
        neuron_dict[i] = {"cluster_id": c, "channel": best_channel, "position": best_position,
                          "amplitudes": cls_amp, "template": best_chan_temp,
                          "neighbor_channels": nbgh_channels, "neighbor_positions": nbgh_postions,
                          "neighbor_templates": nbgh_temps}
    config_dict = dict(zip(channels, positions))
    neuron_data = {0: neuron_dict}
    metadata = {0: config_dict}
    spike_train = list(cluster_agg["spikeTimes"])
    return fs, spike_train, neuron_dict


def sort_template_amplitude(template):
    """
    sort template by amplitude from the largest to the smallest
    :param template: N x M array template array as N for the length of samples,
                     and M for the length of channels
    :return: sorted template index
    """
    assert template.ndim == 2, "Input should be a 2D array; use sort_templates() for higher dimensional data"
    amp = np.max(template, axis=0) - np.min(template, axis=0)
    sorted_idx = np.argsort(amp)[::-1]
    return sorted_idx


def get_best_channel(channels, template):
    assert len(channels) == template.shape[1], "The number of channels does not match to template"
    idx = sort_template_amplitude(template)
    return channels[idx[0]]


def get_best_channel_cluster(clusters, channels, templates):
    """
    find the best channel by sorting templates by amplitude.
    :param clusters:
    :param channels:
    :param templates:
    :return:
    """
    assert len(clusters) == len(templates), "The number of clusters not equal to the number of templates"
    best_channel = dict.fromkeys(clusters)
    for i in range(len(clusters)):
        cls = clusters[i]
        temp = templates[i]
        best_channel[cls] = get_best_channel(channels, temp)
    return best_channel


def get_best_channel_position(channel_position, template):
    idx = sort_template_amplitude(template)
    return channel_position[idx]


def sort_channel_distance(channel_positions, best_channel_position):
    """
    sort channel location by distance to the best channel
    """
    x0, y0 = best_channel_position[0], best_channel_position[1]
    distance = np.empty(len(channel_positions))
    for i in range(len(channel_positions)):
        pos = channel_positions[i]
        distance[i] = (pos[0]-x0)**2 + (pos[1]-y0)**2
    return np.argsort(distance)

