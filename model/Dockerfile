FROM continuumio/miniconda3

ARG model_name=model.tar.gz

WORKDIR /app

COPY . .

ENV PORT 5000
ENV HOST 0.0.0.0

RUN tar -xzf ${model_name} && rm ${model_name}

RUN conda env create -f environment.yml

ENV PATH /opt/conda/envs/poli-bias/bin:$PATH
RUN /bin/bash -c "source activate poli-bias"

EXPOSE 5000

ENTRYPOINT ["python", "server.py", "0.0.0.0", "5000"]
