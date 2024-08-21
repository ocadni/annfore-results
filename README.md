## Reproducing the Results from the Paper:
### Biazzo, I., Braunstein, A., Dall'Asta, L., & Mazza, F. (2022). A Bayesian generative neural network framework for epidemic inference problems. *Scientific Reports, 12*(1), 19673. [https://doi.org/10.1038/s41598-022-20898-x](https://doi.org/10.1038/s41598-022-20898-x)

This repository contains the code and resources necessary to reproduce the results presented in the paper.

### Getting Started

To set up the environment and reproduce the results, follow these steps:

1. **Clone the `epigen` repository:**

   The `epigen` repository contains the core epidemic inference framework. Clone and install it using pip:

   ```bash
   git clone https://github.com/sibyl-team/epigen
   cd epigen
   pip install .
   ```
2.	**Clone the annfore repository:**
  The annfore repository provides the neural network models used in the study. Clone and install it, preferably with the -e option to enable editable mode:

   ```bash
  git clone https://github.com/ocadni/annfore
  cd annfore
  pip install -e .
   ```
3.	**You’re ready to go:**
After completing the installations, you can run the provided scripts and notebooks to reproduce the paper's results.

**Citation**

If you use this code in your research, please cite the following paper:

Biazzo, I., Braunstein, A., Dall’Asta, L., & Mazza, F. (2022). A Bayesian generative neural network framework for epidemic inference problems. Scientific Reports, 12(1), 19673. DOI: [10.1038/s41598-022-20898-x](10.1038/s41598-022-20898-x)
