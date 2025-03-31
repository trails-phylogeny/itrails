CLI for Optimization in iTRAILS
================================

Overview
--------
iTRAILS provides a robust command-line interface (CLI) that streamlines the process of optimizing population parameters for evolutionary inference. The primary command for this task is ``itrails-optimize``, which is designed to integrate easily into your bioinformatics workflows.

Command Structure
-----------------
The ``itrails-optimize`` command accepts between one and three inline arguments:

- **Config File** (required): A YAML configuration file that defines all parameters.
- **--input** (optional): The file path to the MAF alignment file.
- **--output** (optional): The directory and file prefix for the output (formatted as ``directory/prefix``).

Configuration File
------------------
The optimization parameters are defined in a YAML configuration file, which is organized into three sections:

1. **Fixed Parameters**  
   These parameters remain constant during the optimization. For example, the mutation rate can be fixed:
   
   ```yaml
   fixed_parameters:
     mu: 2e-8
   ```
2. **Optimized Parameters**
   Parameters to be optimized are specified as a list in the format ```[starting, minimum, maximum]```. For example:

   ```yaml
   optimized_parameters:  # [starting, min, max]
   N_AB: [50000, 5000, 500000]
   N_ABC: [50000, 5000, 500000]
   t_1: [240000, 24000, 2400000]
   t_2: [40000, 4000, 400000]
   t_3: [800000, 80000, 8000000]
   t_upper: [745069.3855, 74506.9385, 7450693.8556]
   r: [1e-8, 1e-9, 1e-7]
   ```

3. **Settings**
   This section allows you to specify file paths, processing options, and other runtime settings such as the number of cores and the optimization method:

   ```yaml
   settings:
     input_maf: path/to/alignment.maf
     output_prefix: path/to/output_dir/prefix
     n_cpu: 64
     method: "Nelder-Mead"
     species_list: ["hg38", "panTro5", "gorGor5", "ponAbe2"]
     n_int_AB: 3
     n_int_ABC: 3
   ```

Execution and Outputs
------------------
When you run the itrails-optimize command with the appropriate arguments, the tool begins optimizing the parameters to maximize the likelihood of the observed genomic alignment data. During this process, several files are generated:

- **Best Model:**
A YAML file that records the parameter set achieving the highest likelihood. It includes the iteration number, log likelihood, and the optimized parameter values. This file is essential for subsequent analyses, such as Viterbi or posterior decoding.

- **Optimization History:**
A CSV file that logs each iteration of the optimization. It details the parameter values used, their corresponding log likelihood, and the elapsed time for each iteration.

- **Starting Parameters:**
A YAML file that preserves the original configuration, providing a record of the initial parameter space.