Viterbi Decoding
============================

Overview
--------
iTRAILS extends its robust suite of bioinformatics tools to include a Viterbi decoding functionality. This feature is designed to determine the most likely sequence of hidden states (corresponding to gene tree topologies) along a genomic alignment. By invoking the command-line tool ``itrails-viterbi``, you can process an input file—formatted similarly to the Best Model YAML generated during optimization—and obtain outputs that segment the alignment into discrete blocks based on the inferred hidden states.

Command Structure
-----------------
The ``itrails-viterbi`` command accepts a single required argument:

- **Input File** (required): A YAML-formatted file that includes all the necessary parameters for the decoding process. This file is similar in format to the Best Model YAML produced during the optimization step.

Example usage:

.. code-block:: bash

   itrails-viterbi path/to/best_model.yaml

Input File Format
-----------------
The input file for Viterbi decoding mirrors the structure of the Best Model YAML file generated during optimization. It encapsulates all essential parameters required to decode the alignment. Ensure that your input file contains:

- **Population Parameters and Timing Information:** Parameters such as effective population sizes, coalescence times, mutation rates, and recombination rates determined during the optimization phase.
- **Topology Mapping:** Definitions of the gene tree topologies (hidden states) used during decoding.

Execution and Outputs
-----------------------
Upon executing the ``itrails-viterbi`` command, the algorithm processes the genomic alignment and outputs two primary CSV files:

1. **Viterbi Results**  
   This CSV file segments the genomic alignment into discrete blocks where each block is characterized by a uniform hidden state. Each row in the file includes the following columns:
   
   - **Block Index:** A unique index for each alignment block (e.g., 0, 1, 2, ...).
   - **Starting Position:** The first genomic position of the block where a hidden state is assigned.
   - **End Position:** The last genomic position of the block; either the final position within the block or the position immediately before a transition in the hidden state.
   - **Most Likely State:** The hidden state (or topology index) that the Viterbi algorithm determined to be most likely for that block.

2. **Hidden States**  
   This CSV file provides a mapping between the hidden state indices and their corresponding gene tree topologies. Each row includes:
   
   - **State Index:** A unique index (typically ranging from 0 to 26) for each hidden state.
   - **Topology:** The structure of the gene tree. Parentheses ``()`` indicate coalescence within the three-species ancestor, while brackets ``{}`` indicate coalescence within the two-species ancestor.
   - **Interval First Coalescence:** The time interval (in generations) during which the first coalescent event occurs.
   - **Interval Second Coalescence:** The time interval (in generations) during which the second coalescent event occurs.
   - **Shorthand Name:** A compact three-number representation of the topology. The first number represents the general tree topology, while the second and third numbers indicate the time intervals of the coalescence events.

Example Output (Simplified)
-----------------------------
Below is a simplified illustration of what you might see in the output files.

**Viterbi Results (viterbi_results.csv):**

.. code-block:: none

   +-------------+-------------------+--------------+-------------------+
   | Block Index | Starting Position | End Position | Most Likely State |
   +=============+===================+==============+===================+
   | 0           | 1                 | 1500         | 3                 |
   +-------------+-------------------+--------------+-------------------+
   | 1           | 1501              | 3000         | 7                 |
   +-------------+-------------------+--------------+-------------------+
   | 2           | 3001              | 4500         | 3                 |
   +-------------+-------------------+--------------+-------------------+

**Hidden States (hidden_states.csv):**

.. code-block:: none

   +-------------+------------+----------------------------+-----------------------------+----------------+
   | State Index | Topology   | Interval First Coalescence | Interval Second Coalescence | Shorthand Name |
   +=============+============+============================+=============================+================+
   | 0           | (A(B,C))   | 1000                       | 2000                        | 0-1-1          |
   +-------------+------------+----------------------------+-----------------------------+----------------+
   | 1           | ((A,B),C)  | 1500                       | 2500                        | 1-1-2          |
   +-------------+------------+----------------------------+-----------------------------+----------------+
   | ...         | ...        | ...                        | ...                         | ...            |
   +-------------+------------+----------------------------+-----------------------------+----------------+


