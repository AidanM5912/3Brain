import sys
import h5py
import json
import argparse
from dateutil import parser as dateutil_parser  # Rename to avoid conflict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Used for setting the PST timezone
import braingeneers.utils.s3wrangler as wr
from neuroconv.datainterfaces import BiocamRecordingInterface

# New function to extract timestamp from the 3Brain file and set it to PST
def get_time_stamp_3brain(file_path):
    meta_time_stamp = None
    with h5py.File(file_path, "r") as f:
        exp_setting = f["ExperimentSettings"][()][0]  # Get the ExperimentSettings data
        exp_setting = json.loads(exp_setting.decode('utf-8'))  # Decode the JSON data
        ts_org = dateutil_parser.isoparse(exp_setting["ExperimentDateTime"])  # Parse the timestamp
        
        # Set timezone to PST (Pacific Standard Time) using ZoneInfo
        pst_timezone = ZoneInfo("America/Los_Angeles")
        ts_org = ts_org.replace(tzinfo=pst_timezone)  # Apply PST to the timestamp
        
        # Format the timestamp in ISO format with the timezone
        meta_time_stamp = ts_org.isoformat()
        print("3brain Time stamp with PST:", meta_time_stamp)
    return meta_time_stamp

def extract_metadata(file, file_path):
    def decode_attr(attr, default):
        value = file.attrs.get(attr, default)
        return value.decode('utf-8') if isinstance(value, bytes) else value
    
    metadata = {
        "session_description": decode_attr("Description", "No description"),
        "sampling_rate": file.attrs.get("SamplingRate", 1.0),
        # Correctly passing file_path (the string) to get_time_stamp_3brain
        "session_start_time": get_time_stamp_3brain(file_path),  
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
    
    print(f"Extracted metadata: {metadata}")
    return metadata

def convert_brw_to_nwb(input_file, output_file):
    print(f"Starting conversion from {input_file} to {output_file}")

    # Open the input file once and pass it to both functions
    with h5py.File(input_file, 'r') as file:
        # Extract and prepare metadata, passing the file path for the timestamp extraction
        extracted_metadata = extract_metadata(file, input_file)

        # Initialize the BiocamRecordingInterface to handle BRW data
        interface = BiocamRecordingInterface(file_path=input_file, verbose=True)

        # Fetch additional metadata from the interface
        metadata = interface.get_metadata()
    
        # Update NWBFile metadata
        metadata["NWBFile"].update({
            "session_start_time": extracted_metadata["session_start_time"],  # Use the correct timestamp
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
