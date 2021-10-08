import pandas as pd
import numpy as np

num_seed = 10
lambs = []
name_file = "./data/lay_deep_eq_TREE_n_85_d_5_tlim_15_lam_0.35_mu_0_s_0_pe_1_0_trace.gz"
for i in range(1,num_seed):
    name_temp = name_file.replace("s_0", f"s_{i}")
    temp = pd.read_csv(name_temp)
    lambs.append(temp["lamb"].iloc[-1])
lambs = np.array(lambs)
print(f"lamb_mean {lambs.mean()}")
print(lambs)