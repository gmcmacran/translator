
# Repo Overview

This repo follows builds two tensorflow models that translate English to
Spanish.

Models:

- A bidirectional GRU model
- Transformer

Performance Tricks:

- Using the GPU to fit model.
- Using multiple CPU threads to tokenize.
- Using TF dataset to load flat file batch by batch.
- Prefetching flat file.
- Mixed precision.
- Distributed model training if there are multiple GPUs available.
