## Tutorial: Electron affinity of $O_2$ and why diffuse functions matter

**Goals:** 

- learn how to automate calculations with python
- compute the **adiabatic electron affinity** (EA) of oxygen:

$$
\mathrm{O_2 + e^- \rightarrow O_2^-}
$$

- show that **increasing zeta quality alone** (DZ $\rightarrow$ TZ) is *not* enough to converge EA for an anion; **augmentation** (`aug-`, `+`) matters.

We will run a fast DFT method: **UBLYP**.

---

## What you will compute

For each basis set:

- Optimize for **neutral** $ \mathrm{O_2} $ (charge 0, multiplicity 3).
- Optimize for **anion** $ \mathrm{O_2^-} $ (charge −1, multiplicity 2).

Then compute:

- **Electronic EA** (Hartree, then eV):
  $$
  \mathrm{EA} = E(\mathrm{O_2}) - E(\mathrm{O_2^-})
  $$

---

## Prerequisites

### Python install (use links for instructions)

- **Mac / Windows**: Python downloads and installers: `https://www.python.org/downloads/`
- **Linux** (choose one approach):
  - system packages overview: `https://docs.python.org/3/using/unix.html`
  - `pyenv` (common for user-level installs): `https://github.com/pyenv/pyenv#installation`

### Gaussian

Assume Gaussian 16 is installed and the command `g16` works in your terminal.

---

## Project files (already created in this repo)

This tutorial lives in:

- `tutorials/g16-basis-sets/`
  - `inputs/` (starter `.gjf` files)
  - `python/run_basis_ladder.py` (automation script)
  - `requirements.txt` (Python dependencies)
  - `results/` (Gaussian inputs/outputs will be written here by the script)

Starter geometries:

- `tutorials/g16-basis-sets/inputs/O2_triplet_opt_cc-pVDZ.gjf`
- `tutorials/g16-basis-sets/inputs/O2minus_doublet_opt_cc-pVDZ.gjf`

---

## Install Python libraries for automation

From the repo root, run:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r tutorials/g16-basis-sets/requirements.txt
```

Libraries used:

- `cclib`: parse Gaussian outputs (optional in the current script; useful for extensions)
- `numpy`, `pandas`: tabulation
- `matplotlib`: plotting

---

## Basis-set ladder to run (keep it laptop-friendly)

We’ll run this ladder:

1. `cc-pVDZ`
2. `cc-pVTZ`
3. `aug-cc-pVDZ`
4. `aug-cc-pVTZ`

Optional extensions (may be slow on some laptops):

- `cc-pVQZ`, `aug-cc-pVQZ`

---

## Step 1 — sanity-check one manual Gaussian run (recommended)

Run one of the starter jobs to confirm Gaussian works on your machine:

```bash
cd tutorials/g16-basis-sets
g16 inputs/O2_triplet_opt_cc-pVDZ.gjf
```

You should get an output file like:

- `inputs/O2_triplet_opt_cc-pVDZ.log` (or `.out`)

If Gaussian writes output elsewhere on your system, note the behavior and adjust the script accordingly.

---

## Step 2 — run the automated EA workflow

From `tutorials/g16-basis-sets/`, run:

```bash
python3 python/run_basis_ladder.py
```

Read the code to figure out what it does:

- writes all `.gjf` files into `tutorials/g16-basis-sets/results/`
- runs `g16` on each one
- parses:
  - last `SCF Done` energy
  - ZPE (from “Zero-point correction=”)
  - any imaginary frequencies
- prints EA trends and writes a CSV summary:
  - `tutorials/g16-basis-sets/results/ea_o2_basis_ladder_raw.csv`

---

## What trends to look for (the point of the tutorial)

- **Non-augmented Dunning sets** (`cc-pVDZ` $\rightarrow$ `cc-pVTZ`) may change EA, but the anion remains difficult to describe.
- **Augmentation** (`aug-cc-pVDZ`) often causes a **larger shift** than “just going DZ $\rightarrow$ TZ”, because the extra electron needs **diffuse radial flexibility**.

In other words: bigger zeta helps, but **diffuse functions help the *right* part of space**.

---

## Discussion questions

1. Compare EA from `cc-pVDZ` vs `cc-pVTZ`. Is it converging?
2. Compare EA from `cc-pVTZ` vs `aug-cc-pVDZ`. Which change is bigger?

---

## Optional extensions

Modify the script to perform slightly different calculations:

- Repeat with a different functional (e.g., `UPBE`) and compare basis sensitivity.
- Try a Pople ladder:
  - `6-31G(d)` $\rightarrow$ `6-311G(d)` $\rightarrow$ `6-311+G(d)` (diffuse!) $\rightarrow$ `6-311++G(d)` (more diffuse)
- Inspect orbital shapes: does the extra electron density in $ \mathrm{O_2^-} $ look more diffuse with augmentation?

