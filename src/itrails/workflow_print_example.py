def main():
    example_yaml = """\
# Example configuration for itrails-optimize
fixed_parameters:
  n_int_AB: 3
  n_int_ABC: 3
  mu: 2e-8
  method: "Nelder-Mead"
  num_cpu: 64
  species_list: ["hg38", "panTro5", "gorGor5", "ponAbe2"]

optimized_parameters: # [starting, min, max]
  N_AB: [50000, 5000, 500000]
  N_ABC: [50000, 5000, 500000]

  t_A: [240000, 24000, 2400000]
  t_B: [240000, 24000, 2400000]
  t_C: [320000, 32000, 3200000]

  t_2: [40000, 4000, 400000]

  t_3: [800000, 80000, 8000000]
  r: [1e-8, 1e-9, 1e-7]

input:
  maf_alignment: # Path to the MAF alignment file (overwritten by console argument)

output:
  output_name: # Path to the output directory (overwritten by console argument)
"""
    print(example_yaml)


if __name__ == "__main__":
    main()
