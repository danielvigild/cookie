# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# set working directory and install dependencies, the no cache dir is to avoid saving a bunch of extra stuff in the cache that we will never use
WORKDIR /
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy over essential parts of the mnist application
#COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY cookie/ cookie/ 
COPY data/ data/
COPY reports/ reports/



# set or train_model.py as entrypoint, that is the applicaiton we want to run when we execute the image
#ENTRYPOINT ["python", "-u", "cookie/train_model.py", "train --lr 1e-4 --epochs 10"]
CMD ["python", "-u", "cookie/train_model.py", "train", "--lr", "1e-4", "--epochs", "10"]

