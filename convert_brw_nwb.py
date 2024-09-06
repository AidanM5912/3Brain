import sys
import h5py
import argparse
import os
import braingeneers.utils.s3wrangler as wr
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
    print(f"Starting conversion from {input_file} to {output_file}")
    
    # Initialize the BiocamRecordingInterface to handle BRW data
    interface = BiocamRecordingInterface(file_path=input_file, verbose=True)

    # Extract and prepare metadata
    extracted_metadata = extract_metadata(h5py.File(input_file, 'r'))
    metadata = interface.get_metadata()
    
    # Update NWBFile metadata
    metadata["NWBFile"].update({
        "session_start_time": extracted_metadata["session_start_time"].isoformat(),
        "session_description": extracted_metadata["session_description"],
        "identifier": extracted_metadata["session_id"]
    })
    
    # Run the conversion process
    interface.run_conversion(nwbfile_path=output_file, metadata=metadata)

    print(f'Converted {input_file} to {output_file} using Neuroconv')


if __name__ == '__main__':
    print("Starting script...")
    parser = argparse.ArgumentParser(description='Convert BRW data to NWB format.')
    parser.add_argument('input_file', type=str, help='S3 URI for the input .brw file')
    parser.add_argument('output_file', type=str, help='S3 URI for the output .nwb file')
    args = parser.parse_args()

    input_s3 = args.input_file
    output_s3 = args.output_file

    # Create a temporary local directory for processing
    local_input = '/tmp/input_file.brw'
    local_output = '/tmp/output_file.nwb'

    # Download the file from S3 to local using braingeneers' s3wrangler
    print(f"Downloading {input_s3} to {local_input}")
    wr.download(input_s3, local_input)

    # Convert the file locally
    convert_brw_to_nwb(local_input, local_output)

    # Upload the NWB file back to S3 using s3wrangler
    print(f"Uploading {local_output} to {output_s3}")
    wr.upload(local_output, output_s3)

    print("Conversion complete and file uploaded to S3.")
