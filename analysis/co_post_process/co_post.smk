#!python3


rule filter_co_dataset:
  """Filter a crossover dataset according to our key criteria."""
  input:



rule estimate_co_inf_params:
  """Estimate CO-interference parameters in the houseworth-stahl model."""
  input:
    co_filt =  
  output:
     
  script:
    "scripts/est_co_inf.R"
