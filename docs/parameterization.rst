Parameterization of the Model
=============================

iTRAILS leverages the following population parameters in order to infer the underlying gene tree topologies and coalescent event timings (see *Figure iTRAILS_params*).

Population Size Parameters
--------------------------
- **N_AB**: Effective size of the ancestral population between the two most closely related species.
- **N_ABC**: Effective size of the ancestral population between the three species.

Time Parameters (generations)
-----------------------------
- **T1**: Assuming an ultrametric model, time from sample to coalescence of species A and B.
- **T2**: Time between the coalescence of species A and B and the coalescence of AB ancestral with species C.
- **T3**: Time between coalescence of species ABC and coalescence of ABC ancestral with outgroup species.
- **T_A**: Time from sample of species A to coalescence with species B.
- **T_B**: Time from sample of species B to coalescence with species A.
- **T_C**: Time from sample of species C to coalescence with AB ancestral.
- **T_Upper**: Time from the start of the last discretized time interval in ABC ancestral to coalescence with the outgroup.
- **T_Out**: Time from sample of the outgroup to coalescence with ancestral ABC.

Other Parameters
----------------
- **Recombination rate (ρ)**: Number of recombinations per site per generation.
- **Mutation rate (μ)**: Number of mutations per site per generation.

Dependencies and Restrictions
-------------------------------
- **Recombination rate (ρ)** must always be defined either as a fixed parameter or as a parameter to optimize.
- **Mutation rate (μ)** must always be defined as a fixed parameter.
- **T_Upper** and **T3**: At least one of them must be defined (fixed or optimized). If only one is defined, the other is automatically calculated.
- **T_Out**: Can be defined or omitted; if defined, it must be fixed. If omitted, it will be automatically calculated.
- **T_A**, **T_B**, **T_C**, and **T1** should be defined (fixed or optimized) in the following combinations. The non-defined parameters will take values as indicated in the table below.

Relationship Between T_A, T_B, T_C, and T1
============================================

+----------------------+----------------------+----------------------+--------------------------+
| Specified Parameters | Used **T_A**         | Used **T_B**         | Used **T_C**             |
+======================+======================+======================+==========================+
| T_A / T_B / T_C      | T_A                  | T_B                  | T_C                      |
+----------------------+----------------------+----------------------+--------------------------+
| T1 / T_A             | T_A                  | T1                   | T1 + T2                  |
+----------------------+----------------------+----------------------+--------------------------+
| T1 / T_B             | T1                   | T_B                  | T1 + T2                  |
+----------------------+----------------------+----------------------+--------------------------+
| T1 / T_C             | T1                   | T1                   | T_C                      |
+----------------------+----------------------+----------------------+--------------------------+
| T_A / T_B            | T_A                  | T_B                  | (T_A + T_B) / 2 + T2      |
+----------------------+----------------------+----------------------+--------------------------+
| T_A / T_C            | T_A                  | (T_A + T_C - T2) / 2   | T_C                      |
+----------------------+----------------------+----------------------+--------------------------+
| T_B / T_C            | (T_B + T_C - T2) / 2  | T_B                  | T_C                      |
+----------------------+----------------------+----------------------+--------------------------+
| T1                   | T1                   | T1                   | T1 + T2                  |
+----------------------+----------------------+----------------------+--------------------------+
