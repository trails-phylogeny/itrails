# Load functions
from cutpoints import cutpoints_ABC
from optimizer import optimizer
from read_data import maf_parser

# Define fixed parameters
n_int_AB = 3
n_int_ABC = 3
mu = 2e-8
method = "Nelder-Mead"

# Define optimized parameters
N_AB = 25000 * 2 * mu
N_ABC = 25000 * 2 * mu
t_1 = 240000 * mu
t_A = t_1
t_B = t_1
t_2 = 40000 * mu
t_C = t_1 + t_2
t_3 = 800000 * mu
t_upper = t_3 - cutpoints_ABC(n_int_ABC, 1 / N_ABC)[-2]
t_out = t_1 + t_2 + t_3 + 2 * N_ABC
r = 1e-8 / mu

# Define initial parameters
t_init_A = t_A
t_init_B = t_B
t_init_C = t_C
t_init_2 = t_2
t_init_upper = t_upper
N_init_AB = N_AB
N_init_ABC = N_ABC
r_init = r

# Define parameter boundaries as dictionary, with entries being
# 'param_name': [initial_value, lower_bound, upper_bound]
dct = {
    "t_A": [t_init_A, t_A / 10, t_A * 10],
    "t_B": [t_init_B, t_B / 10, t_B * 10],
    "t_C": [t_init_C, t_C / 10, t_C * 10],
    "t_2": [t_init_2, t_2 / 10, t_2 * 10],
    "t_upper": [t_init_upper, t_upper / 10, t_upper * 10],
    "N_AB": [N_init_AB, N_AB / 10, N_AB * 10],
    "N_ABC": [N_init_ABC, N_ABC / 10, N_ABC * 10],
    "r": [r_init, r / 10, r * 10],
}

# Define fixed parameter values
dct2 = {"n_int_AB": n_int_AB, "n_int_ABC": n_int_ABC}

# Define list of species
sp_lst = ["hg38", "panTro5", "gorGor5", "ponAbe2"]
# Read MAF alignment
alignment = maf_parser(
    "/faststorage/project/GenerationInterval/people/dmartinpestana/data/chr1.filtered.region.maf",
    sp_lst,
)

# Run optimization
res = optimizer(
    optim_params=dct,
    fixed_params=dct2,
    V_lst=alignment,
    res_name="optimization.csv",
    method=method,
    header=True,
)
