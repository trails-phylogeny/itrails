Command-Line Interface
======================

itrails provides several command-line entry points:

**itrails-optimize**

.. code-block:: bash

   itrails-optimize --config config.yaml

Optimizes a tree based on ILS model parameters.

**itrails-viterbi**

.. code-block:: bash

   itrails-viterbi --config config.yaml

Runs the Viterbi decoding to find the most likely state sequence.

**itrails-posterior**

.. code-block:: bash

   itrails-posterior --config config.yaml

Computes posterior probabilities across the tree.
