# Qucik script to plot the growth of the background through time

import numpy as np
import matplotlib.pyplot as plt
import scipy


# Use the last year (with decimal) of the analysis
time = [2017.5, ]

# CRN uncorrelated amp, sampled with gamma (log10A)
amp = [-14.7166, ]

# Upper amp error
upper = [0.1432, ]

# Lower amp error
lower = [0.2899, ]







fig, ax = plt.subplots(figsize=(10,10))

ax.errorbar(freq, res, yerr=unc, label = "Timing Residuals", linestyle="", marker=".")
ax.plot(freq, chrom_real, label="Noise realisation")
ax.plot(freq, chrom_fit, label = r"$\beta=4$")
ax.plot(freq, dm_fit, label = r"$\beta=2$")


ax.set_ylabel("Timing Residuals ($\mu$s)", fontsize=15)
ax.set_xlabel("Frequency (MHz)", fontsize=15)
ax.tick_params(labelsize=15)
ax.legend(fontsize=15)

fig.savefig("/fred/oz002/users/mmiles/MPTA_GW/noise_paper_plots_tables/J1017_epoch_CHROM_DM.png")
plt.clf()

