# Neural Network Tutorial: Predicting Molecular Energies with Behler-Parrinello Networks

This tutorial teaches you to build a neural network potential (NNP) that predicts molecular energies from the QM9 dataset using Behler-Parrinello symmetry functions and element-specific neural networks.

## Overview

You will learn:
- How **symmetry functions** encode 3D molecular structure with built-in physical invariances
- How **element-specific neural networks** learn atomic energy contributions
- How the total energy is computed as a sum of atomic contributions
- How to tune hyperparameters to improve model performance

## Prerequisites

### Required Libraries

Install the following Python packages:

```bash
pip install numpy pandas torch scikit-learn ase dscribe tqdm matplotlib
```

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `torch` | Neural network training (PyTorch) |
| `scikit-learn` | Reference energy fitting |
| `ase` | Atomic Simulation Environment (molecular structures) |
| `dscribe` | Behler-Parrinello symmetry functions (ACSF) |
| `tqdm` | Progress bars |
| `matplotlib` | Plotting training curves |

### Data Requirements

The script expects the QM9 dataset as a ZIP file at:
```
../../data/QM9.zip
```

If your data is in a different location, modify the `QM9_ZIP_PATH` variable in the script (around line 1124).

## Running the Tutorial

1. Navigate to the tutorial directory:
   ```bash
   cd tutorials/020-neural-networks
   ```

2. Run the script:
   ```bash
   python NN-tutorial.py
   ```

3. The script will:
   - Extract and load the QM9 dataset
   - Compute Behler-Parrinello symmetry functions (cached for subsequent runs)
   - Train a neural network to predict molecular energies
   - Report training progress and final validation error

**Expected runtime**: ~1-2 minutes with default (suboptimal) settings.

---

## Using AI Assistants (Cursor / VS Code)

We encourage you to use AI coding assistants to help you understand and modify the code. In **Cursor** (or VS Code with Copilot), you have two main modes:

### Ask Mode (Understanding Code)
Use Ask mode to ask questions about the code without making changes. Select code and ask questions like:

- *"Explain what this function does"*
- *"Why do we subtract reference energies from the target?"*
- *"What does the `g2_params` list control in the symmetry functions?"*
- *"How does PyTorch compute gradients through the sum of atomic energies?"*

### Agent Mode (Modifying Code)
Use Agent mode when you want the AI to help you implement changes. Example prompts:

- *"Add code to compute and print the MAE separately for molecules containing nitrogen vs those without"*
- *"Create a scatter plot of predicted vs actual energies for the validation set"*
- *"Modify the training loop to save the best model based on validation loss"*
- *"Add an ensemble of 3 models with different random seeds and compute prediction uncertainty"*

**Tip**: Be specific in your prompts. Instead of "make it better", say "reduce overfitting by adding dropout layers with p=0.1".

---

## Student Tasks

The script is configured with **suboptimal hyperparameters**. Your goal is to improve the model's prediction accuracy (reduce the MAE).

### Task 0: Hyperparameter Tuning

Find and modify the hyperparameters in the **"HYPERPARAMETERS TO TUNE"** section of the script. You can also tune the symmetry function parameters in the same section. The symmetry function cache is automatically invalidated when you change these parameters.

**Goal**: Reduce the validation MAE. Try to achieve chemical accuracy (~1 kcal/mol).

---

### Task 1: Error Analysis by Element Composition

Analyze how prediction errors vary with molecular composition.

**What to do**:
1. After training, group molecules by their element composition
2. Compute the MAE for each group (e.g., "nitrogen-rich", "oxygen-rich", "hydrocarbons only")
3. Create a bar chart showing MAE broken down by composition

**Questions to answer**:
- Which element types are associated with larger errors?
- Does this suggest which element-specific subnetwork needs improvement?

**Hint**: You can use the `all_elements` list to count atoms of each type per molecule.

---

### Task 2: Identifying Hard-to-Predict Molecules

Find the molecules where the model performs worst.

**What to do**:
1. Compute the absolute error for each molecule in the validation set
2. Identify the top 5-10% with the largest errors
3. Analyze their properties: size, element composition, structural features
4. (Optional) If you have UMAP installed, visualize these molecules in symmetry function space

**Questions to answer**:
- Do hard-to-predict molecules share common features?
- Are they unusually large/small, or have unusual element combinations?
- What does this tell you about the model's limitations?

---

### Task 3: Residual Analysis and Error Diagnostics

Create diagnostic plots to understand the model's prediction behavior.

**What to do**:
Create a multi-panel figure with:
1. **Predicted vs. Actual**: Scatter plot with y=x reference line
2. **Residual Histogram**: Distribution of (actual - predicted), should be centered at 0
3. **Residuals vs. Predicted**: Check for patterns (should be random scatter)
4. **Cumulative Error**: What percentage of molecules are within X kcal/mol?

**Questions to answer**:
- Does the model systematically under- or over-predict?
- Is prediction quality uniform across the energy range, or worse for certain values?
- What percentage of predictions are within "chemical accuracy" (1 kcal/mol)?

---

### Task 4: Uncertainty Quantification via Ensemble (Advanced)

Estimate prediction uncertainty by training multiple models.

**What to do**:
1. Train 3-5 models with different random seeds (modify `torch.manual_seed()`)
2. For each validation molecule, compute:
   - Mean prediction across models (ensemble average)
   - Standard deviation across models (uncertainty estimate)
3. Plot: actual error vs. predicted uncertainty

**Questions to answer**:
- Does high ensemble disagreement correlate with high actual error?
- Can you identify molecules where the model "knows it doesn't know"?
- How could uncertainty estimates be useful in practice (e.g., active learning)?

**Hint**: You'll need to modify the training loop to save multiple models or run the script multiple times with different seeds.

---

## Tips

- **Start simple**: First tune `N_EPOCHS` and `LEARNING_RATE` before changing network architecture
- **Watch for overfitting**: If training loss decreases but validation loss increases, you're overfitting
- **Symmetry functions matter**: More G2/G4 functions = richer representation, but also more compute
- **Check the cache**: The symmetry function cache is in `./sf_cache/`. The filename includes a hash of the parameters, so changing parameters creates a new cache file automatically

## Expected Results

With well-tuned hyperparameters, you should be able to achieve:
- MAE < 10 kcal/mol with modest tuning
- MAE < 5 kcal/mol with careful optimization

For comparison, state-of-the-art neural network potentials achieve ~1 kcal/mol on QM9 with much larger training sets and more sophisticated architectures.

## Further Reading

- Behler & Parrinello (2007): "Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces" - The original BP paper
- DScribe documentation: https://singroup.github.io/dscribe/
- PyTorch tutorials: https://pytorch.org/tutorials/
