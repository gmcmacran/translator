
# Repo Overview

This repo follows Deep Learning with Python by Fancois Chollet to build
tensorflow models that translate English to Spanish.

Models:

- A bidirectional GRU model
- Transformer

Performance Tricks:

- Using the GPU to fit model.
- Using multiple CPU threads to tokenize.
- Using TF dataset to load flat file batch by batch.
- Prefetching flat file.
- Mixed precision.
- Distributed model training (if there are multiple GPUs).
