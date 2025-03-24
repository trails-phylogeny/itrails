def main():
    example_yaml = """\
# Example configuration for itrails-optimize
fixed_parameters:
  mu: 2e-8

optimized_parameters: # [starting, min, max]
  N_AB: [50000, 5000, 500000]
  N_ABC: [50000, 5000, 500000]
  t_1: [240000, 24000, 2400000]
  t_2: [40000, 4000, 400000]
  t_3: [800000, 80000, 8000000]
  t_upper: [745069.3855665945, 74506.93855665945, 7450693.855665945]
  r: [1e-8, 1e-9, 1e-7]

settings:
  input_maf: # Path to the MAF alignment file (overwritten by console argument)
  output_name: # Path to the output directory (overwritten by console argument)
  n_cpu: 64
  method: "Nelder-Mead"
  species_list: ["hg38", "panTro5", "gorGor5", "ponAbe2"]
  n_int_AB: 3
  n_int_ABC: 3
"""
    print(example_yaml)


if __name__ == "__main__":
    main()
