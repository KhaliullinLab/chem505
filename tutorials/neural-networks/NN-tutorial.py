#!/usr/bin/env python3
"""
Neural Network Energy Prediction for QM9 Molecules
===================================================

This script loads the QM9 dataset and prepares molecular structures for
neural network-based energy prediction using:
  - ASE (Atomic Simulation Environment) for computing Behler-Parrinello 
    symmetry functions as molecular representations
  - PyTorch for building and training element-specific neural networks

The QM9 dataset contains ~134k small organic molecules (up to 9 heavy atoms: 
C, N, O, F) with quantum mechanical properties computed at the B3LYP/6-31G(2df,p) 
level of DFT.

Our goal: Predict the internal energy at 0K (U0) from molecular structure.

Why Behler-Parrinello symmetry functions?
-----------------------------------------
Unlike fixed-length fingerprints (e.g., Morgan/ECFP), symmetry functions:
  1. Encode 3D geometry explicitly (energy depends on atomic positions)
  2. Are invariant to translation, rotation, and permutation of like atoms
  3. Decompose naturally into atomic contributions (energy is extensive)
  4. Enable element-specific neural networks that capture chemical trends
"""

import os
import warnings
import zipfile
import shutil
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ASE for molecular structure handling
from ase import Atoms

# DScribe for Behler-Parrinello symmetry functions (ACSF)
# DScribe is the standard library for computing atomic descriptors
# Install with: pip install dscribe
from dscribe.descriptors import ACSF

import hashlib
import json

# PyTorch for neural network training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("=" * 70)
print("Neural Network Energy Prediction Tutorial")
print("Using Behler-Parrinello Symmetry Functions + PyTorch")
print("=" * 70)


# =============================================================================
# Section 1: QM9 File Parsing
# =============================================================================
#
# QM9 XYZ files have a specific structure:
#   Line 1:      Number of atoms (N)
#   Line 2:      Properties: gdb tag, A, B, C, μ, α, HOMO, LUMO, gap, R², zpve, U₀, U, H, G, Cᵥ
#   Lines 3-N+2: Atom element, x, y, z, Mulliken partial charge
#   Line N+3:    Harmonic vibrational frequencies (cm⁻¹)
#   Line N+4:    SMILES from GDB-17 and relaxed geometry
#   Line N+5:    InChI from GDB-17 and relaxed geometry
#
# For neural network potentials, we primarily need:
#   - Atomic species (elements)
#   - Atomic coordinates (for computing symmetry functions)
#   - U0: Internal energy at 0K (our prediction target)
# =============================================================================

def parse_float_mathematica(s: str) -> float:
    """
    Parse a float that might be in Mathematica scientific notation.
    
    QM9 dataset has some values like '2.1997*^-6' (Mathematica format)
    instead of '2.1997e-6' (standard Python format).
    """
    # Replace Mathematica's *^ with Python's e for scientific notation
    s = s.replace('*^', 'e')
    return float(s)


def parse_qm9_xyz(filepath: str) -> Optional[Dict]:
    """
    Parse a single QM9 XYZ file and extract molecular structure and properties.
    
    For neural network energy prediction, the key outputs are:
      - atoms: list of element symbols ['C', 'H', 'O', ...]
      - coordinates: atomic positions in Angstroms
      - U0: internal energy at 0K in Hartree (our target property)
    
    Parameters
    ----------
    filepath : str
        Path to the XYZ file
    
    Returns
    -------
    dict or None
        Dictionary containing molecular data, or None if parsing fails
    """
    # Property names in order they appear in line 2 of QM9 XYZ files
    property_names = [
        'tag', 'A', 'B', 'C', 'mu', 'alpha',
        'homo', 'lumo', 'gap', 'R2', 'zpve',
        'U0', 'U', 'H', 'G', 'Cv'
    ]
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Line 0: Number of atoms
        n_atoms = int(lines[0].strip())
        
        # Line 1: Properties (tab-separated)
        # Format: "gdb {tag}\t{A}\t{B}\t..."
        prop_line = lines[1].strip().split('\t')
        
        # Extract tag from "gdb {tag}" format
        tag = int(prop_line[0].split()[1])
        
        # Extract numeric properties
        properties = {'tag': tag}
        for i, value in enumerate(prop_line[1:], start=1):
            if i < len(property_names):
                properties[property_names[i]] = float(value)
        
        # Extract atomic coordinates and elements
        # These are essential for computing symmetry functions
        atoms = []
        coords = []
        charges = []
        
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            atoms.append(parts[0])
            # Some QM9 files use Mathematica notation (*^ instead of e)
            coords.append([
                parse_float_mathematica(parts[1]),
                parse_float_mathematica(parts[2]),
                parse_float_mathematica(parts[3])
            ])
            charges.append(parse_float_mathematica(parts[4]))
        
        properties['atoms'] = atoms
        properties['coordinates'] = np.array(coords)  # Convert to numpy for ASE
        properties['partial_charges'] = charges
        properties['n_atoms'] = n_atoms
        
        # SMILES (useful for visualization/debugging, not for NN training)
        smiles_line = lines[-2].strip().split('\t')
        properties['smiles'] = smiles_line[0]
        
        return properties
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def load_qm9_dataset(data_dir: str, n_molecules: int = 10000) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Load QM9 molecules from XYZ files.
    
    Returns both a DataFrame (for easy analysis) and the raw molecule list
    (which includes coordinates needed for symmetry function computation).
    
    Parameters
    ----------
    data_dir : str
        Directory containing QM9 XYZ files
    n_molecules : int
        Maximum number of molecules to load
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with molecular properties (for analysis/visualization)
    molecules : list of dict
        Raw molecule data including coordinates (for symmetry functions)
    """
    data_path = Path(data_dir)
    xyz_files = sorted(data_path.glob('dsgdb9nsd_*.xyz'))
    
    print(f"Found {len(xyz_files)} XYZ files in {data_dir}")
    print(f"Loading first {n_molecules} molecules...")
    
    molecules = []
    
    for filepath in tqdm(xyz_files[:n_molecules], desc="Parsing XYZ files"):
        mol_data = parse_qm9_xyz(str(filepath))
        if mol_data is not None:
            molecules.append(mol_data)
    
    print(f"Successfully parsed {len(molecules)} molecules")
    
    # Create DataFrame for easy analysis
    # Note: coordinates are stored in the molecules list, not the DataFrame
    df_data = []
    for mol in molecules:
        row = {
            'tag': mol['tag'],
            'smiles': mol['smiles'],
            'n_atoms': mol['n_atoms'],
            'U0': mol['U0'],      # Our target: internal energy at 0K (Hartree)
            'U': mol['U'],        # Internal energy at 298K
            'H': mol['H'],        # Enthalpy at 298K
            'G': mol['G'],        # Free energy at 298K
            'gap': mol['gap'],    # HOMO-LUMO gap (eV)
            'mu': mol['mu'],      # Dipole moment (Debye)
            'alpha': mol['alpha'], # Polarizability
            'Cv': mol['Cv'],      # Heat capacity
            'formula': ''.join(mol['atoms'])
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    return df, molecules


# =============================================================================
# Section 2: Dataset Extraction Utilities
# =============================================================================

def find_qm9_directory(base_dir: Path, zip_stem: str = None) -> Optional[Path]:
    """
    Find the QM9 data directory by checking common naming conventions.
    """
    possible_dirs = [
        base_dir / 'QM9',
        base_dir / 'dsgdb9nsd',
        base_dir / 'qm9',
    ]
    
    if zip_stem and zip_stem.lower() not in ['qm9', 'dsgdb9nsd']:
        possible_dirs.append(base_dir / zip_stem)
    
    for dir_path in possible_dirs:
        if dir_path.exists() and any(dir_path.glob('*.xyz')):
            return dir_path
    
    # Check if xyz files are directly in base_dir
    if any(base_dir.glob('*.xyz')):
        return base_dir
    
    return None


def extract_qm9_dataset(zip_path: str, extract_to: str = None) -> str:
    """
    Extract QM9 dataset from ZIP file.
    
    Parameters
    ----------
    zip_path : str
        Path to the QM9 ZIP file
    extract_to : str, optional
        Directory to extract to. If None, extracts to same directory as ZIP file.
    
    Returns
    -------
    str
        Path to the extracted QM9 directory
    """
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    
    if extract_to is None:
        extract_to = zip_path.parent
    else:
        extract_to = Path(extract_to)
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    zip_stem = zip_path.stem
    existing_dir = find_qm9_directory(extract_to, zip_stem)
    if existing_dir is not None:
        print(f"✓ QM9 dataset already extracted at: {existing_dir}")
        xyz_files = list(existing_dir.glob('*.xyz'))
        print(f"  Found {len(xyz_files)} molecule files (.xyz)")
        return str(existing_dir)
    
    print(f"Extracting QM9 dataset from {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for file in tqdm(file_list, desc="Extracting files"):
                zip_ref.extract(file, extract_to)
        
        qm9_dir = find_qm9_directory(extract_to, zip_stem)
        
        if qm9_dir is None:
            raise RuntimeError(
                f"Extraction completed but no xyz files found in {extract_to}. "
                "The ZIP file may not contain QM9 data in the expected format."
            )
        
        # Rename to standard 'QM9' if needed
        if qm9_dir.name != 'QM9' and qm9_dir != extract_to:
            final_dir = extract_to / 'QM9'
            if final_dir.exists():
                shutil.rmtree(final_dir)
            qm9_dir.rename(final_dir)
            qm9_dir = final_dir
        
        print(f"✓ Extraction complete! Dataset location: {qm9_dir}")
        xyz_files = list(qm9_dir.glob('*.xyz'))
        print(f"  Found {len(xyz_files)} molecule files (.xyz)")
        
        return str(qm9_dir)
    
    except zipfile.BadZipFile:
        raise ValueError(f"Invalid ZIP file: {zip_path}")
    except Exception as e:
        raise RuntimeError(f"Error extracting ZIP file: {e}")


# =============================================================================
# Section 3: Behler-Parrinello Symmetry Functions using DScribe
# =============================================================================
#
# Behler-Parrinello symmetry functions encode the local atomic environment
# as a fixed-length vector. For each atom i, we compute:
#
#   1. RADIAL functions G^rad: Describe distances to neighboring atoms
#      G_i^rad = Σ_j exp(-η(R_ij - R_s)²) · f_c(R_ij)
#
#   2. ANGULAR functions G^ang: Describe bond angles between atom triplets
#      G_i^ang = 2^(1-ζ) Σ_{j,k} (1 + λ·cos(θ_ijk))^ζ · exp(-η(R_ij² + R_ik² + R_jk²)) · f_c(R_ij)·f_c(R_ik)·f_c(R_jk)
#
# Key parameters:
#   - r_cut: Cutoff radius (atoms beyond this distance are ignored)
#   - η (eta): Width of Gaussian functions
#   - R_s: Shift parameter for radial functions
#   - ζ (zeta): Angular resolution parameter
#   - λ (lambda): +1 or -1, determines angular peak position
#
# The result is a feature vector for each atom that:
#   - Is invariant to translation, rotation, and permutation
#   - Captures the local chemical environment
#   - Can be fed into element-specific neural networks
# =============================================================================

def mol_to_ase_atoms(mol: Dict) -> Atoms:
    """
    Convert a molecule dictionary to an ASE Atoms object.
    
    ASE Atoms is the standard data structure for atomic systems, used by
    DScribe and many other atomistic simulation tools.
    
    Parameters
    ----------
    mol : dict
        Molecule dictionary with 'atoms' (element symbols) and 'coordinates'
    
    Returns
    -------
    ase.Atoms
        ASE Atoms object representing the molecule
    """
    return Atoms(
        symbols=mol['atoms'],
        positions=mol['coordinates']
    )


def compute_sf_hash(r_cut: float, g2_params: List, g4_params: List, n_molecules: int) -> str:
    """
    Compute a deterministic hash of symmetry function parameters.
    
    This ensures the cache is invalidated when SF parameters change.
    """
    config = {
        'r_cut': r_cut,
        'g2_params': g2_params,
        'g4_params': g4_params,
        'n_molecules': n_molecules
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def create_acsf_descriptor(species: List[str], r_cut: float, 
                           g2_params: List, g4_params: List) -> ACSF:
    """
    Create an ACSF (Atom-Centered Symmetry Functions) descriptor.
    
    This configures the Behler-Parrinello symmetry functions for molecules.
    
    Parameters
    ----------
    species : list of str
        List of chemical species in the dataset (e.g., ['H', 'C', 'N', 'O', 'F'])
    r_cut : float
        Cutoff radius in Angstroms. Atoms beyond this distance don't contribute.
    g2_params : list of [η, R_s] pairs
        Radial symmetry function parameters.
        η controls width: small η = broad peak, large η = narrow peak
        R_s controls center: where the Gaussian is centered
    g4_params : list of [ζ, λ, η] triplets
        Angular symmetry function parameters.
        ζ controls angular resolution: higher ζ = sharper angular peaks
        λ = +1: peak at 0° (linear), λ = -1: peak at 180°
        η controls radial decay
    
    Returns
    -------
    dscribe.descriptors.ACSF
        Configured ACSF descriptor object
    
    Notes
    -----
    Total feature dimension per atom = 1 + n_G2*n_species + n_G4*n_species_pairs
    """
    acsf = ACSF(
        species=species,
        r_cut=r_cut,
        g2_params=g2_params,
        g4_params=g4_params,
    )
    
    return acsf


def compute_symmetry_functions(
    molecules: List[Dict],
    acsf: ACSF,
    cache_path: Optional[str] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Compute Behler-Parrinello symmetry functions for all molecules.
    
    For each molecule, this produces a feature matrix of shape (n_atoms, n_features)
    where each row is the symmetry function vector for one atom.
    
    Parameters
    ----------
    molecules : list of dict
        List of molecule dictionaries with 'atoms' and 'coordinates'
    acsf : ACSF
        Configured ACSF descriptor from DScribe
    cache_path : str, optional
        Path to cache file. If provided and exists, loads from cache.
        If provided and doesn't exist, computes and saves to cache.
    
    Returns
    -------
    all_features : list of np.ndarray
        Symmetry function matrices, one per molecule. Shape: (n_atoms, n_features)
    all_elements : list of np.ndarray
        Element symbols for each atom, one array per molecule
    species : list of str
        Ordered list of species in the descriptor
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading symmetry functions from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        return cached['features'], cached['elements'], cached['species']
    
    print("\nComputing Behler-Parrinello symmetry functions...")
    print(f"  Cutoff radius: {acsf.r_cut:.1f} Å")
    print(f"  Number of G2 (radial) functions: {len(acsf.g2_params)}")
    print(f"  Number of G4 (angular) functions: {len(acsf.g4_params)}")
    print(f"  Species: {acsf.species}")
    
    all_features = []
    all_elements = []
    
    for mol in tqdm(molecules, desc="Computing symmetry functions"):
        # Convert to ASE Atoms
        atoms = mol_to_ase_atoms(mol)
        
        # Compute ACSF for this molecule
        # Returns shape (n_atoms, n_features)
        features = acsf.create(atoms)
        
        all_features.append(features)
        all_elements.append(np.array(mol['atoms']))
    
    # Get feature dimension
    n_features = all_features[0].shape[1]
    print(f"\n  Feature vector dimension: {n_features} per atom")
    
    # Save to cache
    if cache_path:
        print(f"Saving symmetry functions to cache: {cache_path}")
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'features': all_features,
                'elements': all_elements,
                'species': acsf.species
            }, f)
    
    return all_features, all_elements, acsf.species


def compute_reference_energies(molecules: List[Dict], species: List[str]) -> Dict[str, float]:
    """
    Compute per-element reference energies using linear regression.
    
    The idea: Total energy ≈ Σ_elem (count[elem] × E_ref[elem])
    
    This accounts for the fact that a molecule with more carbon atoms
    naturally has a lower (more negative) energy. By subtracting these
    reference energies, the neural network only needs to learn the 
    smaller deviations due to bonding and geometry.
    
    Parameters
    ----------
    molecules : list of dict
        Molecule data with 'U0' and 'atoms'
    species : list of str
        List of element symbols
    
    Returns
    -------
    dict
        Maps element symbol to reference energy (Hartree)
    """
    from sklearn.linear_model import LinearRegression
    
    # Build design matrix: counts of each element per molecule
    X = np.zeros((len(molecules), len(species)))
    y = np.array([mol['U0'] for mol in molecules])
    
    for i, mol in enumerate(molecules):
        for atom in mol['atoms']:
            elem_idx = species.index(atom)
            X[i, elem_idx] += 1
    
    # Fit linear regression (no intercept - pure atomic contributions)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    
    # Extract reference energies
    ref_energies = {elem: reg.coef_[i] for i, elem in enumerate(species)}
    
    return ref_energies


def prepare_training_data(
    molecules: List[Dict],
    all_features: List[np.ndarray],
    all_elements: List[np.ndarray],
    species: List[str]
) -> Dict:
    """
    Prepare training data organized by element type.
    
    For Behler-Parrinello neural network potentials, we train separate networks
    for each element. This function organizes the data accordingly.
    
    The total molecular energy is the sum of atomic contributions:
        E_total = Σ_i E_i(G_i)
    
    where E_i is predicted by the element-specific network from symmetry 
    function vector G_i.
    
    IMPORTANT: We subtract reference atomic energies from the total energy.
    This normalization is crucial because raw energies vary by hundreds of
    Hartree depending on molecule size. After normalization, the target
    becomes the "atomization energy" or deviation from reference.
    
    Parameters
    ----------
    molecules : list of dict
        Original molecule data (for U0 targets)
    all_features : list of np.ndarray
        Symmetry function matrices from compute_symmetry_functions()
    all_elements : list of np.ndarray
        Element arrays from compute_symmetry_functions()
    species : list of str
        Ordered species list
    
    Returns
    -------
    data : dict
        Dictionary containing:
        - 'targets': array of NORMALIZED U0 values (n_molecules,)
        - 'targets_raw': array of original U0 values
        - 'ref_energies': dict of per-element reference energies
        - 'features_by_element': dict mapping element -> list of feature vectors
        - 'mol_indices_by_element': dict mapping element -> list of (mol_idx, atom_idx)
        - 'n_atoms_per_mol': array of atom counts
        - 'species': list of species
        - 'n_features': feature dimension
    """
    print("\nPreparing training data organized by element...")
    
    # -------------------------------------------------------------------------
    # Compute and subtract reference energies
    # -------------------------------------------------------------------------
    # This is CRITICAL for good performance. Raw U0 values vary by hundreds
    # of Hartree depending on molecule size. By fitting linear reference
    # energies per element, we transform the target to be much smaller.
    print("\n  Computing per-element reference energies...")
    ref_energies = compute_reference_energies(molecules, species)
    
    for elem, e_ref in ref_energies.items():
        print(f"    {elem}: {e_ref:.4f} Ha")
    
    # Extract and normalize target energies
    targets_raw = np.array([mol['U0'] for mol in molecules])
    
    # Compute reference energy for each molecule
    ref_per_mol = np.zeros(len(molecules))
    for i, mol in enumerate(molecules):
        for atom in mol['atoms']:
            ref_per_mol[i] += ref_energies[atom]
    
    # Normalized targets = raw - reference
    targets = targets_raw - ref_per_mol
    
    print(f"\n  Raw U0 range: [{targets_raw.min():.2f}, {targets_raw.max():.2f}] Ha")
    print(f"  Normalized range: [{targets.min():.4f}, {targets.max():.4f}] Ha")
    print(f"  Normalized std: {targets.std():.4f} Ha")
    
    # Organize features by element
    # This allows training element-specific networks
    features_by_element = {elem: [] for elem in species}
    mol_indices_by_element = {elem: [] for elem in species}
    
    for mol_idx, (features, elements) in enumerate(zip(all_features, all_elements)):
        for atom_idx, elem in enumerate(elements):
            features_by_element[elem].append(features[atom_idx])
            mol_indices_by_element[elem].append((mol_idx, atom_idx))
    
    # Convert lists to arrays
    for elem in species:
        if features_by_element[elem]:
            features_by_element[elem] = np.vstack(features_by_element[elem])
        else:
            features_by_element[elem] = np.array([])
    
    # Count atoms per molecule (needed for summing atomic contributions)
    n_atoms_per_mol = np.array([len(mol['atoms']) for mol in molecules])
    
    # Print statistics
    print(f"\n  Number of molecules: {len(targets)}")
    print(f"  Target (U0) range: [{targets.min():.2f}, {targets.max():.2f}] Hartree")
    print("\n  Atoms per element:")
    for elem in species:
        count = len(mol_indices_by_element[elem])
        print(f"    {elem}: {count:,} atoms")
    
    return {
        'targets': targets,               # Normalized targets (for training)
        'targets_raw': targets_raw,       # Original U0 values
        'ref_energies': ref_energies,     # Per-element reference energies
        'features_by_element': features_by_element,
        'mol_indices_by_element': mol_indices_by_element,
        'n_atoms_per_mol': n_atoms_per_mol,
        'species': species,
        'n_features': all_features[0].shape[1]
    }


# =============================================================================
# Section 4: PyTorch Neural Network for Energy Prediction
# =============================================================================
#
# The Behler-Parrinello neural network architecture:
#   1. Each element has its own MLP (multi-layer perceptron)
#   2. Each MLP maps: symmetry_functions → atomic_energy_contribution
#   3. Total molecular energy = sum of all atomic contributions
#   4. We only have total energies for training, but gradients flow back
#      through the sum to train each element-specific network
#
# Key insight: Even though we don't have atomic energy labels, the networks
# learn consistent atomic contributions because atoms in similar chemical
# environments (similar symmetry functions) should contribute similar energies.
# =============================================================================

class ElementNetwork(nn.Module):
    """
    A small MLP that predicts atomic energy contribution for one element type.
    
    Architecture: Input → Hidden1 → Hidden2 → Output (single scalar)
    """
    def __init__(self, n_features: int, hidden_sizes: List[int] = [64, 32]):
        super().__init__()
        
        layers = []
        prev_size = n_features
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())  # Smooth activation, works well for potentials
            prev_size = hidden_size
        
        # Final layer outputs a single scalar (atomic energy contribution)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        x: (n_atoms, n_features) symmetry function vectors
        Returns: (n_atoms, 1) atomic energy contributions
        """
        return self.network(x)


class BehlerParrinelloNN(nn.Module):
    """
    Behler-Parrinello Neural Network Potential.
    
    Contains one ElementNetwork for each chemical species. The total energy
    is the sum of atomic contributions from all atoms.
    """
    def __init__(self, n_features: int, species: List[str], hidden_sizes: List[int] = [64, 32]):
        super().__init__()
        
        self.species = species
        self.n_features = n_features
        
        # Create one network per element
        # nn.ModuleDict ensures PyTorch tracks all sub-networks
        self.element_networks = nn.ModuleDict({
            elem: ElementNetwork(n_features, hidden_sizes)
            for elem in species
        })
    
    def forward(self, features_by_element: Dict[str, torch.Tensor], 
                mol_indices: Dict[str, List[Tuple[int, int]]],
                n_molecules: int) -> torch.Tensor:
        """
        Compute total energy for each molecule.
        
        Parameters
        ----------
        features_by_element : dict
            Maps element symbol to tensor of shape (n_atoms_of_element, n_features)
        mol_indices : dict
            Maps element symbol to list of (molecule_idx, atom_idx) tuples
        n_molecules : int
            Total number of molecules in the batch
        
        Returns
        -------
        torch.Tensor
            Predicted total energies, shape (n_molecules,)
        """
        # Initialize molecular energies to zero
        device = next(self.parameters()).device
        molecular_energies = torch.zeros(n_molecules, device=device)
        
        # Process each element type
        for elem in self.species:
            if elem not in features_by_element or len(features_by_element[elem]) == 0:
                continue
            
            features = features_by_element[elem]
            indices = mol_indices[elem]
            
            # Get atomic energy contributions from this element's network
            atomic_energies = self.element_networks[elem](features).squeeze(-1)
            
            # Add each atom's contribution to its molecule's total energy
            for atom_idx, (mol_idx, _) in enumerate(indices):
                molecular_energies[mol_idx] += atomic_energies[atom_idx]
        
        return molecular_energies


class QM9Dataset(Dataset):
    """
    PyTorch Dataset for QM9 molecules with precomputed symmetry functions.
    
    For efficiency, we store all data and return indices for batching.
    The actual batching logic handles the element-wise organization.
    """
    def __init__(self, targets: np.ndarray, features_list: List[np.ndarray], 
                 elements_list: List[np.ndarray]):
        """
        Parameters
        ----------
        targets : np.ndarray
            Target energies (U0) for each molecule
        features_list : list of np.ndarray
            Symmetry function matrices, one per molecule
        elements_list : list of np.ndarray
            Element arrays, one per molecule
        """
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.features_list = features_list
        self.elements_list = elements_list
        self.n_molecules = len(targets)
    
    def __len__(self):
        return self.n_molecules
    
    def __getitem__(self, idx):
        return {
            'idx': idx,
            'target': self.targets[idx],
            'features': self.features_list[idx],
            'elements': self.elements_list[idx]
        }


def collate_molecules(batch: List[Dict], species: List[str]) -> Dict:
    """
    Custom collate function to organize a batch of molecules by element.
    
    This is where we reorganize data from per-molecule to per-element format,
    which is needed for efficient forward passes through element networks.
    """
    indices = [item['idx'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    
    # Organize features by element
    features_by_element = {elem: [] for elem in species}
    mol_indices_by_element = {elem: [] for elem in species}
    
    for batch_mol_idx, item in enumerate(batch):
        features = item['features']
        elements = item['elements']
        
        for atom_idx, elem in enumerate(elements):
            features_by_element[elem].append(features[atom_idx])
            mol_indices_by_element[elem].append((batch_mol_idx, atom_idx))
    
    # Convert lists to tensors
    for elem in species:
        if features_by_element[elem]:
            features_by_element[elem] = torch.tensor(
                np.array(features_by_element[elem]), dtype=torch.float32
            )
        else:
            features_by_element[elem] = torch.tensor([], dtype=torch.float32)
    
    return {
        'indices': indices,
        'targets': targets,
        'features_by_element': features_by_element,
        'mol_indices_by_element': mol_indices_by_element,
        'n_molecules': len(batch)
    }


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """
    Train for one epoch.
    
    Returns
    -------
    float
        Average loss over the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Move data to device
        targets = batch['targets'].to(device)
        features_by_element = {
            elem: feat.to(device) if len(feat) > 0 else feat
            for elem, feat in batch['features_by_element'].items()
        }
        
        # Forward pass
        predictions = model(
            features_by_element,
            batch['mol_indices_by_element'],
            batch['n_molecules']
        )
        
        # Compute loss (MSE)
        loss = nn.functional.mse_loss(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict:
    """
    Evaluate model on a dataset.
    
    Returns
    -------
    dict
        Contains 'mse', 'rmse', 'mae', and arrays of predictions/targets
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            targets = batch['targets'].to(device)
            features_by_element = {
                elem: feat.to(device) if len(feat) > 0 else feat
                for elem, feat in batch['features_by_element'].items()
            }
            
            predictions = model(
                features_by_element,
                batch['mol_indices_by_element'],
                batch['n_molecules']
            )
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    
    mse = nn.functional.mse_loss(predictions, targets).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'predictions': predictions.numpy(),
        'targets': targets.numpy()
    }


def train_model(training_data: Dict, all_features: List[np.ndarray], 
                all_elements: List[np.ndarray],
                n_epochs: int = 100, batch_size: int = 32, 
                learning_rate: float = 1e-3, hidden_sizes: List[int] = [64, 32],
                train_split: float = 0.8, device: str = 'cpu') -> Dict:
    """
    Train a Behler-Parrinello neural network on QM9 data.
    
    Parameters
    ----------
    training_data : dict
        Output from prepare_training_data()
    all_features : list of np.ndarray
        Symmetry function matrices per molecule
    all_elements : list of np.ndarray
        Element arrays per molecule
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimizer
    hidden_sizes : list of int
        Hidden layer sizes for element networks
    train_split : float
        Fraction of data for training (rest for validation)
    device : str
        'cpu' or 'cuda'
    
    Returns
    -------
    dict
        Contains trained model, training history, and evaluation results
    """
    device = torch.device(device)
    
    # -------------------------------------------------------------------------
    # Prepare train/validation split
    # -------------------------------------------------------------------------
    n_molecules = len(training_data['targets'])
    n_train = int(n_molecules * train_split)
    
    # Shuffle indices
    indices = np.random.permutation(n_molecules)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print("\nData split:")
    print(f"  Training:   {len(train_indices)} molecules")
    print(f"  Validation: {len(val_indices)} molecules")
    
    # Create datasets
    train_dataset = QM9Dataset(
        training_data['targets'][train_indices],
        [all_features[i] for i in train_indices],
        [all_elements[i] for i in train_indices]
    )
    
    val_dataset = QM9Dataset(
        training_data['targets'][val_indices],
        [all_features[i] for i in val_indices],
        [all_elements[i] for i in val_indices]
    )
    
    # Create dataloaders with custom collate function
    species = training_data['species']
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_molecules(b, species)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_molecules(b, species)
    )
    
    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    model = BehlerParrinelloNN(
        n_features=training_data['n_features'],
        species=species,
        hidden_sizes=hidden_sizes
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print("\nModel created:")
    print(f"  Element networks: {species}")
    print(f"  Hidden layers: {hidden_sizes}")
    print(f"  Total parameters: {n_params:,}")
    
    # -------------------------------------------------------------------------
    # Training setup
    # -------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_results = evaluate(model, val_loader, device)
        val_loss = val_results['mse']
        val_mae = val_results['mae']
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, "
                  f"Val MAE = {val_mae:.4f} Ha")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("-" * 60)
    print("\nFinal evaluation on validation set:")
    final_results = evaluate(model, val_loader, device)
    print(f"  MSE:  {final_results['mse']:.6f} Ha²")
    print(f"  RMSE: {final_results['rmse']:.6f} Ha")
    print(f"  MAE:  {final_results['mae']:.6f} Ha")
    
    # Convert to more intuitive units (kcal/mol)
    ha_to_kcal = 627.509  # 1 Hartree = 627.509 kcal/mol
    print(f"\n  MAE:  {final_results['mae'] * ha_to_kcal:.2f} kcal/mol")
    print(f"  RMSE: {final_results['rmse'] * ha_to_kcal:.2f} kcal/mol")
    
    return {
        'model': model,
        'history': history,
        'final_results': final_results,
        'train_indices': train_indices,
        'val_indices': val_indices
    }


# =============================================================================
# Section 5: Main Execution
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Adjust this path to point to your QM9 ZIP file
    QM9_ZIP_PATH = '../../data/QM9.zip'
    
    # -------------------------------------------------------------------------
    # HYPERPARAMETERS TO TUNE
    # -------------------------------------------------------------------------
    # The current values are intentionally suboptimal. Your task is to find
    # better hyperparameters that reduce the prediction error (MAE).
    #
    # -------------------------------------------------------------------------
    
    # Number of molecules to load from QM9
    N_MOLECULES = 500
    
    # Number of training epochs
    N_EPOCHS = 20
    
    # Learning rate for the Adam optimizer
    LEARNING_RATE = 0.01
    
    # Hidden layer sizes for element-specific networks [layer1, layer2]
    HIDDEN_SIZES = [16, 8]
    
    # Fraction of data used for training (rest is validation)
    TRAIN_SPLIT = 0.6
    
    # -------------------------------------------------------------------------
    # SYMMETRY FUNCTION PARAMETERS (Advanced)
    # -------------------------------------------------------------------------
    # TIP: You can also improve results by tuning the symmetry functions!
    # The cache is automatically invalidated when these parameters change
    # (a hash of the parameters is included in the cache filename).
    # 
    # G2 (radial): Each [η, R_s] pair creates radial features
    #   - η (eta): width of Gaussian (small=broad, large=narrow)
    #   - R_s: center distance in Angstroms
    #
    # G4 (angular): Each [ζ, λ, η] triplet creates angular features
    #   - ζ (zeta): angular sharpness (higher=sharper peaks)
    #   - λ (lambda): +1 favors 0° angles, -1 favors 180°
    #   - η (eta): radial decay rate
    #
    # -------------------------------------------------------------------------
    
    # Cutoff radius for symmetry functions (Angstroms)
    R_CUT = 4.0
    
    # G2 radial symmetry function parameters [η, R_s]
    # Each [η, R_s] pair creates one radial function per element type
    # η controls width: small η = broad peak, large η = narrow peak
    # R_s controls center: where the Gaussian is centered
    G2_PARAMS = [
        [0.1, 0.0],
        [0.1, 2.0],
        [0.5, 1.0],
        [0.5, 2.0],
    ]
    
    # G4 angular symmetry function parameters [ζ, λ, η]
    # Each [ζ, λ, η] triplet creates one angular function per element pair
    # ζ controls angular resolution: higher ζ = sharper angular peaks
    # λ = +1: peak at 0° (linear), λ = -1: peak at 180°
    # η controls radial decay
    G4_PARAMS = [
        [1, 1, 0.01],
        [2, 1, 0.01],
    ]
    
    # -------------------------------------------------------------------------
    # Fixed configuration (do not change)
    # -------------------------------------------------------------------------
    # Elements present in QM9 dataset
    QM9_SPECIES = ['H', 'C', 'N', 'O', 'F']
    
    # =========================================================================
    # Step 1: Load QM9 Dataset
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 1: Extracting QM9 dataset")
    print("-" * 70)
    QM9_DATA_DIR = extract_qm9_dataset(QM9_ZIP_PATH)
    
    print("\n" + "-" * 70)
    print("Step 2: Loading molecular structures and properties")
    print("-" * 70)
    df_qm9, molecules_raw = load_qm9_dataset(QM9_DATA_DIR, n_molecules=N_MOLECULES)
    
    # -------------------------------------------------------------------------
    # Display dataset summary
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 3: Dataset Summary")
    print("-" * 70)
    print(f"\nLoaded {len(df_qm9)} molecules")
    
    print("\n--- Target Property: U0 (Internal Energy at 0K) ---")
    print("  Unit: Hartree")
    print(f"  Min:  {df_qm9['U0'].min():.4f}")
    print(f"  Max:  {df_qm9['U0'].max():.4f}")
    print(f"  Mean: {df_qm9['U0'].mean():.4f}")
    print(f"  Std:  {df_qm9['U0'].std():.4f}")
    
    print("\n--- Molecule Size Distribution ---")
    print(df_qm9['n_atoms'].value_counts().sort_index())
    
    # =========================================================================
    # Step 2: Compute Behler-Parrinello Symmetry Functions
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 4: Setting up Behler-Parrinello Symmetry Functions")
    print("-" * 70)
    
    # Create the ACSF descriptor with the configured parameters
    acsf = create_acsf_descriptor(
        species=QM9_SPECIES, 
        r_cut=R_CUT,
        g2_params=G2_PARAMS,
        g4_params=G4_PARAMS
    )
    
    print("\nACSF descriptor configuration:")
    print(f"  Species: {acsf.species}")
    print(f"  Cutoff radius: {R_CUT} Å")
    print(f"  Number of G2 (radial) parameter sets: {len(acsf.g2_params)}")
    print(f"  Number of G4 (angular) parameter sets: {len(acsf.g4_params)}")
    
    # Compute symmetry functions for all molecules
    print("\n" + "-" * 70)
    print("Step 5: Computing Symmetry Functions")
    print("-" * 70)
    
    # Generate cache path with hash of SF parameters (auto-invalidates when params change)
    sf_hash = compute_sf_hash(R_CUT, G2_PARAMS, G4_PARAMS, N_MOLECULES)
    SF_CACHE_PATH = f'./sf_cache/symmetry_functions_{sf_hash}.pkl'
    print(f"  Cache file: {SF_CACHE_PATH}")
    
    all_features, all_elements, species = compute_symmetry_functions(
        molecules_raw, 
        acsf,
        cache_path=SF_CACHE_PATH
    )
    
    # -------------------------------------------------------------------------
    # Visualize a sample symmetry function vector
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 6: Symmetry Function Examples")
    print("-" * 70)
    
    sample_idx = 0
    sample_mol = molecules_raw[sample_idx]
    sample_features = all_features[sample_idx]
    
    print(f"\nSample molecule (tag={sample_mol['tag']}):")
    print(f"  Formula: {''.join(sample_mol['atoms'])}")
    print(f"  Number of atoms: {len(sample_mol['atoms'])}")
    print(f"  Symmetry function matrix shape: {sample_features.shape}")
    print(f"    → {sample_features.shape[0]} atoms × {sample_features.shape[1]} features")
    
    print(f"\n  First atom ({sample_mol['atoms'][0]}) symmetry function vector:")
    print(f"    Min:  {sample_features[0].min():.4f}")
    print(f"    Max:  {sample_features[0].max():.4f}")
    print(f"    Mean: {sample_features[0].mean():.4f}")
    print(f"    First 10 values: {sample_features[0][:10]}")
    
    # =========================================================================
    # Step 3: Prepare Data for Neural Network Training
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 7: Preparing Data for Element-Specific Networks")
    print("-" * 70)
    
    training_data = prepare_training_data(
        molecules_raw,
        all_features,
        all_elements,
        species
    )
    
    # -------------------------------------------------------------------------
    # Summary of data preparation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"""
Summary:
  - Loaded {len(molecules_raw)} molecules from QM9
  - Computed {training_data['n_features']}-dimensional symmetry functions per atom
  - Target property: U0 (internal energy at 0K)
  - Species: {training_data['species']}
""")
    
    # =========================================================================
    # Step 4: Train Neural Network
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 8: Training Behler-Parrinello Neural Network")
    print("-" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Training configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {DEVICE}")
    
    # Train the model using the hyperparameters defined above
    results = train_model(
        training_data=training_data,
        all_features=all_features,
        all_elements=all_elements,
        n_epochs=N_EPOCHS,
        batch_size=32,              # Fixed: molecules per batch
        learning_rate=LEARNING_RATE,
        hidden_sizes=HIDDEN_SIZES,
        train_split=TRAIN_SPLIT,
        device=DEVICE
    )
    
    # =========================================================================
    # Step 5: Analyze Results
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 9: Training Results Analysis")
    print("-" * 70)
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        axes[0].plot(results['history']['train_loss'], label='Train Loss')
        axes[0].plot(results['history']['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss (Ha²)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Parity plot
        preds = results['final_results']['predictions']
        targets = results['final_results']['targets']
        axes[1].scatter(targets, preds, alpha=0.3, s=10)
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        axes[1].set_xlabel('True U0 (Ha)')
        axes[1].set_ylabel('Predicted U0 (Ha)')
        axes[1].set_title('Parity Plot (Validation Set)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150)
        print("\nTraining plots saved to 'training_results.png'")
        plt.close()
        
    except ImportError:
        print("\nMatplotlib not available, skipping plots")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    ha_to_kcal = 627.509  # Conversion factor
    ha_to_ev = 27.211     # Conversion factor
    
    mae_ha = results['final_results']['mae']
    rmse_ha = results['final_results']['rmse']
    
    print(f"""
Final Model Performance (Validation Set):
  ─────────────────────────────────────────────────────────
  Note: Errors are on NORMALIZED energies (after subtracting
  per-element reference energies). This is what matters for
  predicting energy differences, reaction energies, etc.
  ─────────────────────────────────────────────────────────
  
  MAE:  {mae_ha:.6f} Ha = {mae_ha * ha_to_kcal:.2f} kcal/mol = {mae_ha * ha_to_ev * 1000:.1f} meV
  RMSE: {rmse_ha:.6f} Ha = {rmse_ha * ha_to_kcal:.2f} kcal/mol = {rmse_ha * ha_to_ev * 1000:.1f} meV

Reference Energies Used:""")
    for elem, e_ref in training_data['ref_energies'].items():
        print(f"    {elem}: {e_ref:.4f} Ha")
    
    print("""
Key Takeaways:
  - Behler-Parrinello NNs learn atomic energy contributions from total energies
  - Energy normalization (subtracting reference energies) is CRITICAL
  - Symmetry functions encode 3D structure with built-in physical invariances
  - Element-specific networks capture chemical trends
  - This architecture scales linearly with system size
  
For comparison, chemical accuracy is ~1 kcal/mol (~43 meV).
State-of-the-art NNPs achieve ~1-10 meV/atom on QM9 with larger datasets.
""")
