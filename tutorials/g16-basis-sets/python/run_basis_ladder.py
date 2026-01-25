from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class Species:
    name: str
    charge: int
    multiplicity: int
    # XYZ lines without charge/mult line
    xyz_lines: tuple[str, ...]


O2 = Species(
    name="O2_triplet",
    charge=0,
    multiplicity=3,
    xyz_lines=(
        "O   0.000000   0.000000   0.000000",
        "O   0.000000   0.000000   1.210000",
    ),
)

O2M = Species(
    name="O2minus_doublet",
    charge=-1,
    multiplicity=2,
    xyz_lines=(
        "O   0.000000   0.000000   0.000000",
        "O   0.000000   0.000000   1.330000",
    ),
)


def write_gjf(
    *,
    out_path: Path,
    chk_name: str,
    route: str,
    title: str,
    charge: int,
    multiplicity: int,
    xyz_lines: Iterable[str],
    mem: str = "2GB",
    nproc: int = 4,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"%Mem={mem}",
            f"%NProcShared={nproc}",
            f"%Chk={chk_name}",
            route,
            "",
            title,
            "",
            f"{charge} {multiplicity}",
            *xyz_lines,
            "",
            "",  # Gaussian requires a blank line at the end
        ]
    )
    out_path.write_text(text, encoding="utf-8")


def run_g16(input_path: Path) -> Path:
    """
    Run Gaussian on an input file. Assumes `g16` is on PATH.

    Returns the path to the produced log/out file.
    """
    subprocess.run(["g16", str(input_path)], check=True)

    candidates = [
        input_path.with_suffix(".log"),
        input_path.with_suffix(".out"),
        input_path.with_suffix(".lst"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Gaussian run finished but no output found for {input_path.name}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


_SCF_RE = re.compile(r"SCF Done:\s+E\([A-Z-]+\)\s+=\s+(-?\d+\.\d+)")


def parse_scf_energy_hartree(text: str) -> float:
    matches = _SCF_RE.findall(text)
    if not matches:
        raise ValueError("Could not find 'SCF Done:' energy in output.")
    return float(matches[-1])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    inputs_dir = root / "inputs"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Basis ladder: show that "bigger zeta" without augmentation is not enough,
    # then show augmentation makes a bigger difference for the anion.
    basis_sets = [
        "cc-pVDZ",
        "cc-pVTZ",
        # Optional extension (comment in if laptops allow)
        # "cc-pVQZ",
        "aug-cc-pVDZ",
        "aug-cc-pVTZ",
        # "aug-cc-pVQZ",
    ]

    # Use unrestricted DFT consistently for the open-shell neutral (triplet) and anion (doublet).
    method = "UBLYP"

    rows: list[dict[str, float | str]] = []

    for basis in basis_sets:
        for sp in (O2, O2M):
            stem = f"{sp.name}_{method}_{basis}".replace("/", "_")
            gjf = results_dir / f"{stem}.gjf"

            write_gjf(
                out_path=gjf,
                chk_name=f"{stem}.chk",
                route=f"#P {method}/{basis} Opt SCF=Tight Integral=UltraFine NoSymm",
                title=f"{sp.name} {method}/{basis} Opt",
                charge=sp.charge,
                multiplicity=sp.multiplicity,
                xyz_lines=sp.xyz_lines,
            )

            print(f"[run] {gjf.name}")
            log_path = run_g16(gjf)
            text = log_path.read_text(errors="replace")
            e_h = parse_scf_energy_hartree(text)

            rows.append(
                {
                    "basis": basis,
                    "species": sp.name,
                    "E_hartree": e_h,
                }
            )

        # compute EA after both species are done
        e_neu = next(r["E_hartree"] for r in rows if r["basis"] == basis and r["species"] == O2.name)  # type: ignore[no-any-return]
        e_an = next(r["E_hartree"] for r in rows if r["basis"] == basis and r["species"] == O2M.name)  # type: ignore[no-any-return]
        ea_h = float(e_neu) - float(e_an)
        ea_ev = ea_h * HARTREE_TO_EV

        print(f"[EA] basis={basis:12s}  EA(elec)={ea_ev: .3f} eV")

    # Write a simple CSV summary for plotting/analysis.
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(results_dir / "ea_o2_basis_ladder_raw.csv", index=False)
        print(f"[ok] wrote {results_dir / 'ea_o2_basis_ladder_raw.csv'}")
    except Exception as e:
        print(f"[warn] could not write CSV with pandas: {e}")


if __name__ == "__main__":
    main()

