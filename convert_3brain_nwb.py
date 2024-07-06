import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import h5py # type: ignore
from pynwb import NWBFile, NWBHDF5IO
from pynwb.base import TimeSeries
from pynwb.ecephys import ElectricalSeries
import numpy as np

#load raw time series data for brw and bxr files 
def load_spike_data(file, file_extension):
    well_id = '/Well_A1'
    if file_extension == 'brw':
        timestamps = np.array(file['TOC'][:, 0])
        data = np.zeros((len(timestamps), len(file[well_id + '/EventsBasedSparseRaw'])), dtype=np.int16)
        DecodeEventBasedRawData(file, data, well_id, 0, len(timestamps))
    elif file_extension == 'bxr':
        spike_times = file[well_id + '/SpikeTimes'][:]
        spike_forms = file[well_id + '/SpikeForms'][:]
        timestamps = spike_times / file.attrs.get("SamplingRate", 1.0)
        data = spike_forms
    return data, timestamps

#extract metadata
def extract_metadata(file):
    def decode_attr(attr, default):
        value = file.attrs.get(attr, default)
        return value.decode('utf-8') if isinstance(value, bytes) else value
    metadata = {
        "description": decode_attr("Description", "No description"),
        "sampling_rate": file.attrs.get("SamplingRate", 1.0),
        "session_start_time": file.attrs.get("ExperimentDateTimeUtc", None),
        "experiment_type": decode_attr("ExperimentType", "Unknown"),
        "guid": decode_attr("GUID", "No GUID"),
        "source_guid": decode_attr("SourceGUID", "No SourceGUID"),
        "max_analog_value": file.attrs.get("MaxAnalogValue", None),
        "max_digital_value": file.attrs.get("MaxDigitalValue", None),
        "min_analog_value": file.attrs.get("MinAnalogValue", None),
        "min_digital_value": file.attrs.get("MinDigitalValue", None),
        "plate_model": file.attrs.get("PlateModel", None),
        "version": file.attrs.get("Version", None)
    }
    if metadata["session_start_time"] is not None:
        # convert timestamp to datetime (assuming a Unix timestamp in nanoseconds)
        metadata["session_start_time"] = datetime.fromtimestamp(metadata["session_start_time"] / 1e9, tz=timezone.utc)
    else:
        # default session start time if not found in metadata
        # this needs to be something different but for now its the current time and date
        metadata["session_start_time"] = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))
    return metadata

#adding this from the script arianna sent
def DecodeEventBasedRawData(file, data, wellID, startFrame, numFrames):
# collect the TOCs
    toc = np.array(file['TOC'])
    eventsToc = np.array(file[wellID + '/EventsBasedSparseRawTOC'])
    # from the given start position and duration in frames, localize the corresponding event positions
    # using the TOC
    tocStartIdx = np.searchsorted(toc[:, 1], startFrame)
    tocEndIdx = min(np.searchsorted(toc[:, 1], startFrame + numFrames, side='right')
    + 1, len(toc) - 1)
    eventsStartPosition = eventsToc[tocStartIdx]
    eventsEndPosition = eventsToc[tocEndIdx]
    # decode all data for the given well ID and time interval
    binaryData = file[wellID + '/EventsBasedSparseRaw'][eventsStartPosition:eventsEndPosition]
    binaryDataLength = len(binaryData)
    pos = 0
    while pos < binaryDataLength:
        chIdx = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        chDataLength = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        chDataPos = pos
    while pos < chDataPos + chDataLength:
        fromInclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
        pos += 8
        toExclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
        pos += 8
        rangeDataPos = pos
        for j in range(fromInclusive, toExclusive):
            if j >= startFrame + numFrames:
                break
            if j >= startFrame:
                data[chIdx][j - startFrame] = int.from_bytes(
                binaryData[rangeDataPos:rangeDataPos + 2], byteorder='little', signed=True)
                rangeDataPos += 2
                pos += (toExclusive - fromInclusive) * 2

def convert_biocam_to_nwb(input_file, output_file):
    with h5py.File(input_file, 'r') as file:
        metadata = extract_metadata(file)
        well_id = 'WellA1'
        
        # Determine file extension
        file_extension = input_file.split('.')[-1]

        data, timestamps = load_spike_data(file, file_extension)
        
        # Prepare NWB file
        nwbfile = NWBFile(
            session_description=metadata["description"],
            identifier=metadata["guid"],
            session_start_time=metadata["session_start_time"]
        )
        
        # Create TimeSeries object for spike times and spike forms
        if file_extension == 'brw':
            timeseries = TimeSeries(
                name='EventsBasedSparseRaw',
                data=data,
                unit='arbitrary',
                timestamps=timestamps
            )
            nwbfile.add_acquisition(timeseries)
        elif file_extension == 'bxr':
            spike_timeseries = TimeSeries(
                name='SpikeTimes',
                data=timestamps,
                unit='seconds',
                timestamps=timestamps
            )
            spike_forms_series = ElectricalSeries(
                name='SpikeForms',
                data=data,
                electrodes=np.arange(data.shape[1]),
                starting_time=0.0,
                rate=metadata["sampling_rate"],
                conversion=1.0,
                resolution=-1.0,
                unit='microvolts'
            )
            nwbfile.add_acquisition(spike_timeseries)
            nwbfile.add_acquisition(spike_forms_series)
        
        # Add source GUID to the NWB file general info
        nwbfile.source_script_file_name = metadata["source_guid"]
        
        # Write the NWB file
        with NWBHDF5IO(output_file, 'w') as io:
            io.write(nwbfile)
        
        print(f'Converted {input_file} to {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Biocam data to NWB format.')
    parser.add_argument('input_file', type=str, help='Path to the input .brw or .bxr file')
    parser.add_argument('output_file', type=str, help='Path to the output .nwb file')
    args = parser.parse_args()
    
    convert_biocam_to_nwb(args.input_file, args.output_file)