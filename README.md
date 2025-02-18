# multi_fidelity_gpflow

An implementation of multi-fidelity emulators from emukit using GPFlow

Packages

## Demo: For testing Forrester function

```bash
python tests/test_forrest.py
```

![](images/Figure_1.png)
![](images/Figure_2.png)
![](images/Figure_3.png)

## Demo: For testing Ho-Bird-Shelton 2021 50LF-3HF DM only power spectra

### 1) Multi-bin

```bash
# you need pip install pytest
pytest tests/test_ho2021_multibin.py
```

Also check `notebooks/demo: matter power.ipynb`

### 2) Single-bin Sparse GP

This is making use of multi-output kernel in GPFlow, so is expected to be faster than looping k bins in the old version.

```bash
# you need pip install pytest
pytest tests/test_ho2021_singlebin.py # This runs quite slowly
```

Also check `notebooks/demo: matter power single bin.ipynb`

### 3) Latent Sparse GP multi-output

This is making use of linear-coregionalization kernel in GPflow, ideally will do the factor analysis (linear low-rank decomposition) inside the GP training. So reduce the num output bins GP to low-rank num of latent GPs to train.

Check `notebooks/demo: matter power latent inference.ipynb`

## Demo: For testing z=0 Goku matter power spectra


Check `notebooks/demo: goku power spectra.ipynb`. It takes multi-bin, single-bin, and latent inference all together. But I haven't tested the save/load function. So probably need to read the gpflow doc to know how to save.

Summary of the time costs by each method (on my 2023 Macbook Pro M1 chip, CPU)


| Method               | Estimated Time |
|----------------------|---------------|
| Multi-bin Inference | 137.63 seconds |
| Single-bin Inference | 1724.09 seconds |
| Latent Inference    | 539.54 seconds |

### Pred/Exact in test set

1) Multi-bin: 

![](images/test_goku/pred_exact_multibin.png)

2) Single-bin:

![](images/test_goku/pred_exact_singlebin.png)

3) Latent inference:

![](images/test_goku/pred_exact_latentinference.png)

### Diagnostic plots

A few ways you can debug/interpret your inference results. First thing you can do is looking at the single-bin inference's lengthscales and scale parameters. Here we only look at scale parameters rho(k)

![](images/test_goku/rho_singlebin.png)

There is an apparent power suppression at higher k due to LF's resolution limitation, so higher rho to scale up the suppression.

Another plot is to view the mixing matrix W learned from the latent inference, this way we can know how much different output bins (k modes in this case) are correlated, so we can reduce the number of hyperparameters to train.

![](images/test_goku/learned_W_matrix.png)

---
_Last updated: [2025/02/17]_

