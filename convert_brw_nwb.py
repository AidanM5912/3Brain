import sys
import h5py
import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from neuroconv.datainterfaces import BiocamRecordingInterface

def extract_metadata(file):
    def decode_attr(attr, default):
        value = file.attrs.get(attr, default)
        return value.decode('utf-8') if isinstance(value, bytes) else value
    
    metadata = {
        "session_description": decode_attr("Description", "No description"),
        "sampling_rate": file.attrs.get("SamplingRate", 1.0),
        "session_start_time": file.attrs.get("ExperimentDateTimeUtc", None),
        "session_id": decode_attr("GUID", "No GUID"),
        "max_analog_value": file.attrs.get("MaxAnalogValue", None),
        "max_digital_value": file.attrs.get("MaxDigitalValue", None),
        "min_analog_value": file.attrs.get("MinAnalogValue", None),
        "min_digital_value": file.attrs.get("MinDigitalValue", None),
        "experiment_type": decode_attr("ExperimentType", "Unknown"),
        "source_guid": decode_attr("SourceGUID", "No SourceGUID"),
        "plate_model": file.attrs.get("PlateModel", None),
        "version": file.attrs.get("Version", None)
    }
    
    if metadata["session_start_time"] is not None:
        # convert timestamp to datetime (assuming a Unix timestamp in nanoseconds)
        metadata["session_start_time"] = datetime.fromtimestamp(metadata["session_start_time"] / 1e9, tz=timezone.utc)
    else:
        # default session start time if not found in metadata
        metadata["session_start_time"] = datetime(2024, 1, 30, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))
    
    print(f"Extracted metadata: {metadata}")
    return metadata

def convert_brw_to_nwb(input_file, output_file):
    interface = BiocamRecordingInterface(file_path=input_file, verbose=True)

    # extract and prepare metadata
    extracted_metadata = extract_metadata(h5py.File(input_file, 'r'))
    metadata = interface.get_metadata()
    
    # update NWBFile metadata
    metadata["NWBFile"].update({
        "session_start_time": extracted_metadata["session_start_time"].isoformat(),
        "session_description": extracted_metadata["session_description"],
        "identifier": extracted_metadata["session_id"]
    })
    # convert
    interface.run_conversion(nwbfile_path=output_file, metadata=metadata)

    print(f'Converted {input_file} to {output_file} using Neuroconv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BRW data to NWB format.')
    parser.add_argument('input_file', type=str, help='Path to the input .brw file')
    parser.add_argument('output_file', type=str, help='Path to the output .nwb file')
    args = parser.parse_args()

    file_extension = args.input_file.split('.')[-1].lower()

    if file_extension == 'brw':
        convert_brw_to_nwb(args.input_file, args.output_file)
    else:
        print(f"Unsupported file extension: {file_extension}. Please provide a .brw file.")
