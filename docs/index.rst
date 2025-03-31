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

- Using PyPi

.. code-block:: bash

    pip install itrails

- Using conda

.. code-block:: bash

    conda install conda-forge::itrails

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
   This landing page covers installation and basic usage. For further details on each command-line function, please refer to their individual documentation pages. Happy analyzing!

- **Source Code:** Access the codebase or contribute to the project on `GitHub <https://github.com/trails-phylogeny/itrails>`_.


.. toctree::
   :maxdepth: 2
   :hidden:

   parameterization
   cli_optm
   cli_vit
   cli_post
   api
