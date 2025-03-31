iTRAILS Documentation
=====================

Welcome to the iTRAILS documentation landing page. iTRAILS is a command-line tool designed to infer population parameters and reconstruct evolutionary histories from genomic data using a coalescent hidden Markov model framework. This landing page provides a quick start with installation instructions and an overview of basic usage. Detailed guides for each CLI function are available on their own dedicated pages.

Overview
--------

iTRAILS builds upon the original TRAILS framework by offering a more programmatic and user-friendly implementation for:

- **Parameter Optimization:** Estimate key population parameters by maximizing the likelihood of genomic alignment data.
- **Gene Tree Decoding:** Use both Viterbi and posterior decoding to determine the most likely sequence of hidden states (gene tree topologies) across genomic segments.

Installation
------------

iTRAILS is available both on PyPi and via conda. Choose the installation method that best fits your environment.

### Using PyPi

.. code-block:: bash

    pip install itrails

### Using conda

.. code-block:: bash

    conda install -c bioconda itrails

Basic Usage
-----------

iTRAILS consists of several command-line functions. Here’s a brief overview of the core functionalities:

- **itrails-optimize:**  
  Optimizes critical population parameters using a YAML configuration file along with input genomic data.  
  _Example usage:_

  .. code-block:: bash

      itrails-optimize config.yaml --input path/to/alignment.maf --output path/to/output_prefix

- **itrails-viterbi:**  
  Applies Viterbi decoding to infer the most likely sequence of gene tree topologies along the genome.

- **itrails-posterior:**  
  Computes the posterior probability distribution of hidden states at every alignment position.

.. note::
   Detailed documentation for each CLI function is available on separate pages.

Getting Started with Parameter Optimization
---------------------------------------------

Begin by creating a YAML configuration file that specifies fixed and optimized parameters. For example:

.. code-block:: yaml

    fixed_parameters:
      mu: 2e-8

    optimized_parameters:
      N_AB: [50000, 5000, 500000]
      N_ABC: [50000, 5000, 500000]
      t_1: [240000, 24000, 2400000]
      t_2: [40000, 4000, 400000]
      t_3: [800000, 80000, 8000000]
      t_upper: [745069.3855, 74506.9385, 7450693.8556]
      r: [1e-8, 1e-9, 1e-7]

    settings:
      input_maf: path/to/alignment.maf
      output_prefix: path/to/output_dir/prefix
      n_cpu: 64
      method: "Nelder-Mead"
      species_list: ["hg38", "panTro5", "gorGor5", "ponAbe2"]
      n_int_AB: 3
      n_int_ABC: 3

Run the optimization with:

.. code-block:: bash

    itrails-optimize config.yaml --input path/to/alignment.maf --output path/to/output_dir/prefix

Upon completion, the tool produces a **Best Model** file containing the optimized parameters. This file serves as the configuration for subsequent decoding analyses.

Additional Resources
--------------------

- **Full Documentation:** For more comprehensive instructions, usage examples, and API reference, please visit the `iTRAILS Read the Docs <https://itrails.readthedocs.io/en/stable/>`_.
- **Source Code:** Access the codebase or contribute to the project on `GitHub <https://github.com/trails-phylogeny/itrails>`_.

This landing page covers installation and basic usage. For further details on each command-line function, please refer to their individual documentation pages. Happy analyzing!


.. toctree::
   :maxdepth: 2
   :hidden:

   cli
   api
