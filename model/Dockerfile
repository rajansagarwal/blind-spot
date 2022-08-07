# Use miniconda3 image
FROM continuumio/miniconda3

# Set default model argument
ARG model_name=model.tar.gz

# Specify working directory
WORKDIR /app

# Copy over files
COPY . .

# Set required environment variables for
#   google cloud
ENV PORT 5000
ENV HOST 0.0.0.0

# Extract desired model and remove archive
RUN tar -xzf ${model_name} && rm ${model_name}

# Create the environment
RUN conda env create -f environment.yml

# Activate conda environment
ENV PATH /opt/conda/envs/poli-bias/bin:$PATH
RUN /bin/bash -c "source activate poli-bias"

# Expose the correct port
EXPOSE 5000

# The code to run when container is started:
ENTRYPOINT ["python", "server.py", "0.0.0.0", "5000"]
