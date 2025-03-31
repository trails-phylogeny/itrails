# iTRAILS

iTRAILS is a command-line tool designed to infer population parameters and reconstruct evolutionary histories from genomic data using a coalescent hidden Markov model framework. It offers functionalities such as parameter optimization, Viterbi decoding, and posterior decoding, all wrapped in a user-friendly interface.

## Features

- **Parameter Optimization:** Optimize critical population parameters from genomic alignment data.
- **Gene Tree Decoding:** Infer gene tree topologies across the genome using Viterbi and posterior decoding.
- **Command Line Interface (CLI):** Easily integrate iTRAILS into your bioinformatics workflows.
- **Config File Support:** Customize parameters via YAML configuration files without altering the source code.

## Installation

Install iTRAILS using PyPi:

```bash
pip install itrails
```
Or with conda:
```bash
conda install conda-forge::itrails
```
## Quick Start

1. Create a YAML configuration file defining fixed and optimized parameters.
2. Run the parameter optimization with:

```bash
itrails-optimize config.yaml --input path/to/alignment.maf --output path/to/output/output_prefix
```

3. Run the parameter optimization with: Use the generated Best Model configuration file to perform gene tree decoding with:
  - Viterbi Decoding: ```itrails-viterbi```
  - Posterior Decoding: ```itrails-posterior```

## Documentation

For more detailed instructions, usage examples, and API references, please visit our full documentation at:
[Read The Docs - iTRAILS](https://itrails.readthedocs.io/en/stable/)