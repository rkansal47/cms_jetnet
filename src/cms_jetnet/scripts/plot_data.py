import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

jets = pd.read_parquet("../../../../data/QCD_Pt_1000to1400/")

plot_dir = "../../../plots/"
import os

os.system(f"mkdir -p {plot_dir}")

plot_vars = {
    "RecoJet_mass": [0, 1000, 100],
    "RecoJet_pt": [0, 2000, 100],
    "RecoJet_phi": [-4, 4, 100],
    "RecoJet_eta": [-4, 4, 100],
    "GenJet_mass": [0, 1000, 100],
    "GenJet_pt": [0, 2000, 100],
    "GenJet_phi": [-4, 4, 100],
    "GenJet_eta": [-4, 4, 100],
}


for var, bins in plot_vars.items():
    _ = plt.hist(jets[var].values.reshape(-1), np.linspace(*bins), histtype="step")
    plt.xlabel(var)
    plt.ylabel("# jets")
    plt.savefig(f"{plot_dir}/{var}.pdf")
    plt.show()


jets
