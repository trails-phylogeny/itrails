Posterior Decoding in iTRAILS
=============================

Overview
--------
iTRAILS enhances its decoding capabilities with posterior decoding functionality. Instead of returning a single most likely sequence of hidden states, posterior decoding computes the probability distribution over all possible hidden states (gene tree topologies) at every position along the genomic alignment. This approach provides a richer, probabilistic insight into the underlying evolutionary processes.

Command Structure
-----------------
The posterior decoding is performed using the ``itrails-posterior`` command. This command requires a single input file formatted similarly to the Best Model YAML file produced during optimization.

Example usage:

.. code-block:: bash

   itrails-posterior path/to/best_model.yaml

Input File Format
-----------------
The input file must contain all the necessary parameters for decoding and is formatted similarly to the Best Model YAML file generated in the optimization step. It should include:

- **Population Parameters and Timing Information:** Such as effective population sizes, coalescence times, mutation rates, and recombination rates.
- **Topology Mapping:** Definitions of the gene tree topologies (hidden states) that are considered during decoding.

Execution and Outputs
-----------------------
Upon execution of the ``itrails-posterior`` command, the posterior decoding algorithm processes the genomic alignment and generates two primary CSV files:

1. **Posterior Results**  
   This CSV file provides the probability distribution over hidden states for each nucleotide position in the alignment blocks. Each row corresponds to a single position and includes the following columns:

   - **Block Index:** A unique index for each alignment block.
   - **Position Index:** A unique index for the nucleotide position within the block.
   - **End Position:** The final position in the block where a hidden state is assigned; this could be the end of the block or the position just before a transition occurs.
   - **State N Probability:** A series of columns, each representing the probability for a specific hidden state (topology).

2. **Hidden States**  
   This CSV file maps the hidden state indices to their corresponding gene tree topologies. It is identical in structure to the Hidden States file produced during Viterbi decoding. For a detailed explanation, please refer to the Viterbi decoding subsection.

Example Output (Simplified)
-----------------------------
Below is a simplified example illustrating the expected outputs.

**Posterior Results (posterior_results.csv):**

.. code-block:: none

   +-------------+-----------------+--------------+----------------------+----------------------+----------------------+
   | Block Index | Position Index  | End Position | State 0 Probability  | State 1 Probability  | ... State N Probability |
   +=============+=================+==============+======================+======================+======================+
   | 0           | 1               | 1500         | 0.05                 | 0.95                 | ... 0.00              |
   +-------------+-----------------+--------------+----------------------+----------------------+----------------------+
   | 0           | 2               | 1500         | 0.10                 | 0.90                 | ... 0.00              |
   +-------------+-----------------+--------------+----------------------+----------------------+----------------------+
   | ...         | ...             | ...          | ...                  | ...                  | ...                  |
   +-------------+-----------------+--------------+----------------------+----------------------+----------------------+

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
