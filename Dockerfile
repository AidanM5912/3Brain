FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# run convert_biocam_to_nwb.py when the container launches
ENTRYPOINT ["python", "/app/convert_3brain_nwb.py"]
# defualt to using a brw file as input
# this can be overriden in the k8s job file. just specify the data input path in the args section
CMD ["/data/input_file.brw", "/data/output_file.nwb"]
