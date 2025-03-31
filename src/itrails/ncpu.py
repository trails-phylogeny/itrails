import multiprocessing as mp
import os

# Get the allocated CPU count from SLURM (or fall back to the total CPU count)
ALLOCATED_CPUS = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE", mp.cpu_count()))

# Default: use all allocated CPUs unless overridden
N_CPU = ALLOCATED_CPUS


def update_n_cpu(user_requested):
    """
    Update the global N_CPU based on the user-requested number of CPUs. This function sets N_CPU to the minimum of the user request and ALLOCATED_CPUS. It also updates environment variables used by numerical libraries.

    :param user_requested: Number of CPU cores requested by the user.
    :type user_requested: int.
    """
    global N_CPU
    try:
        requested = int(user_requested)
    except (TypeError, ValueError):
        requested = ALLOCATED_CPUS  # if invalid, use default
    N_CPU = min(requested, ALLOCATED_CPUS)

    # Update environment variables
    os.environ["OMP_NUM_THREADS"] = str(N_CPU)
    os.environ["MKL_NUM_THREADS"] = str(N_CPU)
    os.environ["NUMEXPR_NUM_THREADS"] = str(N_CPU)
    os.environ["RAYON_NUM_THREADS"] = str(N_CPU)
    os.environ["RAY_NUM_THREADS"] = str(N_CPU)

    print(
        f"Using {N_CPU} CPU cores (requested: {requested}, allocated: {ALLOCATED_CPUS})."
    )
