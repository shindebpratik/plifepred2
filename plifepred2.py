#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLifePred2 standalone prediction engine.

Supported prediction modes
--------------------------
1. natural   : amino-acid sequence; all five modification flags are zero
2. modified  : amino-acid sequence plus selected modification flags
3. smiles    : SMILES -> RDKit -> vendored Mordred-community 2D descriptors
4. structure : PDB -> RDKit-derived SMILES -> vendored Mordred-community descriptors

All model and feature files are resolved relative to this script's project
folder. The script can therefore be launched from another working directory.

Prediction is the default operation. Add --design to run natural single-residue
scanning or all 32 modification-flag combinations. Input counts are not capped:
sequence records, SMILES records, and PDB input files can be supplied in bulk.
Design mode retains the original 15-50 aa peptide-length range.

Expected project layout
-----------------------
PLifePred2/
    plifepred2_master_v4.py
    pfeature/comp/pfeature_comp.py       # required for sequence modes
    models/
        Sequence_model.sav
        RFR_combined_model.pkl
        Struture_model.pkl
        sequence_features.tsv
        smile_features.csv
        mordred-community/mordred/__init__.py

The Pfeature directory must contain the complete Pfeature installation,
including its accompanying Data/resources, not only pfeature_comp.py.

The target transformations are preserved from the supplied server version:
- natural/modified sequence: y = log10(t), so t = 10**y
- SMILES:                  y = -log10(t), so t = 10**(-y)
- structure:               y = log10(t), so t = 10**y

This file is an inference script; it does not train or modify model files.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


# ---------------------------------------------------------------------------
# Project paths. Keep the directory names synchronized with the repository.
# Paths are anchored to this script rather than the shell's current directory.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DEFAULT_SEQUENCE_MODEL = MODEL_DIR / "Sequence_model.sav"
DEFAULT_SEQUENCE_FEATURES = MODEL_DIR / "sequence_features.tsv"
DEFAULT_SMILES_MODEL = MODEL_DIR / "RFR_combined_model.pkl"
DEFAULT_SMILES_FEATURES = MODEL_DIR / "smile_features.csv"
DEFAULT_STRUCTURE_MODEL = MODEL_DIR / "Struture_model.pkl"
DEFAULT_MORDRED_DIR = MODEL_DIR / "mordred-community"
DEFAULT_PFEATURE = BASE_DIR / "comp" / "pfeature_comp.py"

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
MOD_COLUMNS = ["d", "ct", "nt", "cyc", "ptm"]
MOD_FLAG_COLUMNS = MOD_COLUMNS.copy()
SEQUENCE_FEATURE_TYPES = "AAC,DPC,PCP,RRI,DDR,SEP,SER,SPC,CTC,CeTD,PAAC,APAAC,QSO"

# Design-mode peptide length limits inherited from the original design module.
DESIGN_MIN_SEQUENCE_LENGTH = 15
DESIGN_MAX_SEQUENCE_LENGTH = 50
DESIGN_FLAG_COMBINATIONS = tuple(itertools.product((0, 1), repeat=len(MOD_COLUMNS)))


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def log(message: object) -> None:
    print(f"[INFO] {message}", file=sys.stderr)


def require_file(path: Path | str, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate.resolve()


def read_feature_names(
    path: Path | str,
    expected_column: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> list[str]:
    """Read an ordered feature list from a CSV/TSV file."""
    feature_path = require_file(path, "Selected-feature file")
    if delimiter is None:
        delimiter = "\t" if feature_path.suffix.lower() in {".tsv", ".tab"} else ","

    table = pd.read_csv(feature_path, sep=delimiter)
    if expected_column:
        if expected_column not in table.columns:
            raise ValueError(
                f"Feature file {feature_path} must contain a '{expected_column}' column; "
                f"found columns: {table.columns.tolist()}"
            )
        names = table[expected_column].dropna().astype(str).str.strip().tolist()
    else:
        names = [str(column).strip() for column in table.columns]

    names = [name for name in names if name]
    if not names:
        raise ValueError(f"No feature names were found in {feature_path}.")
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate feature names in {feature_path}: {duplicates[:10]}")
    return names


def _make_unique_ids(ids: Sequence[str]) -> list[str]:
    """Preserve input IDs while ensuring output IDs are unique."""
    seen: set[str] = set()
    unique: list[str] = []
    for index, raw_id in enumerate(ids, start=1):
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_id).strip()) or f"Sequence_{index}"
        candidate = base
        suffix = 2
        while candidate in seen:
            candidate = f"{base}_{suffix}"
            suffix += 1
        seen.add(candidate)
        unique.append(candidate)
    return unique


def read_sequence_input(path: Path | str) -> list[tuple[str, str]]:
    """Read FASTA records or one plain peptide sequence per non-empty line.

    Only the 20 standard amino-acid symbols are accepted, matching the
    original PLifePred2 model input assumptions.
    """
    input_path = require_file(path, "Sequence input")
    text = input_path.read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("Sequence input is empty.")

    lines = text.splitlines()
    has_fasta_headers = any(line.lstrip().startswith(">") for line in lines)
    records: list[tuple[str, str]] = []

    if has_fasta_headers:
        current_id: Optional[str] = None
        current_sequence: list[str] = []
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_sequence)))
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header on line {line_number}.")
                current_id = header.split()[0]
                current_sequence = []
            else:
                if current_id is None:
                    raise ValueError(
                        f"Sequence text appears before the first FASTA header (line {line_number})."
                    )
                current_sequence.append(re.sub(r"\s+", "", line))
        if current_id is not None:
            records.append((current_id, "".join(current_sequence)))
    else:
        for index, line in enumerate(lines, start=1):
            sequence = re.sub(r"\s+", "", line).upper()
            if sequence:
                records.append((f"Sequence_{index}", sequence))

    if not records:
        raise ValueError("No peptide sequences were found in the input file.")

    ids = _make_unique_ids([record[0] for record in records])
    validated: list[tuple[str, str]] = []
    for (old_id, sequence), seq_id in zip(records, ids):
        sequence = re.sub(r"\s+", "", sequence).upper()
        if not sequence:
            raise ValueError(f"Sequence '{old_id}' is empty.")
        invalid = sorted(set(sequence) - AA20)
        if invalid:
            raise ValueError(
                f"Sequence '{old_id}' contains unsupported symbol(s): {', '.join(invalid)}. "
                "Use the 20 standard amino acids: ACDEFGHIKLMNPQRSTVWY."
            )
        validated.append((seq_id, sequence))
    return validated


def parse_modifications(value: Optional[str]) -> list[str]:
    """Parse modification names or legacy numeric labels."""
    if not value or value.strip().lower() in {"none", "natural", "0mods", "no"}:
        return []

    numeric_mapping = {"0": "d", "1": "ct", "2": "nt", "3": "cyc", "4": "ptm"}
    selected: list[str] = []
    for item in value.split(","):
        item = item.strip().lower()
        item = numeric_mapping.get(item, item)
        if item not in MOD_COLUMNS:
            raise ValueError(
                f"Unknown modification '{item}'. Choose from d, ct, nt, cyc, ptm "
                "(or legacy numeric values 0-4)."
            )
        if item not in selected:
            selected.append(item)
    return selected


# ---------------------------------------------------------------------------
# Pfeature sequence descriptor generation
# ---------------------------------------------------------------------------
def run_pfeature(
    input_fasta: Path | str,
    output_csv: Path | str,
    feature_types: str = SEQUENCE_FEATURE_TYPES,
    pfeature_script: Path | str = DEFAULT_PFEATURE,
    python_executable: Optional[str] = None,
) -> pd.DataFrame:
    """Run the bundled/local Pfeature script and load its CSV output.

    The full Pfeature directory (including its Data resources) must be present
    under BASE_DIR/pfeature. The active Python interpreter is used by default.
    """
    pfeature_path = require_file(pfeature_script, "Pfeature script")
    input_path = require_file(input_fasta, "Pfeature FASTA input")
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    executable = python_executable or sys.executable

    command = [
        executable,
        str(pfeature_path),
        "-i", str(input_path),
        "-o", str(output_path),
        "-j", feature_types,
        "-wd", str(output_path.parent),
    ]
    log(f"Running Pfeature using {executable}: {feature_types}")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Pfeature failed with exit code {completed.returncode}. "
            f"Check that its dependencies and Data directory are installed.\n{details[-5000:]}"
        )
    if not output_path.is_file():
        raise RuntimeError(f"Pfeature completed but did not create {output_path}.")

    try:
        feature_df = pd.read_csv(output_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read Pfeature output {output_path}: {exc}") from exc
    if feature_df.empty:
        raise RuntimeError("Pfeature generated an empty feature matrix.")
    return feature_df


# ---------------------------------------------------------------------------
# Model loading and feature alignment
# ---------------------------------------------------------------------------
def load_model(path: Path | str):
    """Load a trusted local estimator, including legacy custom-transformer pickles.

    The structure model was pickled with the class reference
    ``default_regression_models.DescriptorPreprocessor``.  The small
    default_regression_models.py compatibility module in BASE_DIR restores that
    import path without shipping the original model-training script.
    """
    model_path = require_file(path, "Model file")
    log(f"Loading model: {model_path}")

    # Make the project directory importable even when this module is invoked
    # through an entry point or from a different working directory.
    base_dir_string = str(BASE_DIR)
    if base_dir_string not in sys.path:
        sys.path.insert(0, base_dir_string)

    # Import this only for the model known to require the legacy transformer.
    # If another model later uses this class, this import is harmless to retain.
    if model_path.name == DEFAULT_STRUCTURE_MODEL.name:
        try:
            importlib.import_module("default_regression_models")
        except Exception as exc:
            raise RuntimeError(
                "The structure model requires the inference compatibility module "
                "default_regression_models.py beside plifepred2_master_v4.py. "
                f"Could not import it: {exc}"
            ) from exc

    try:
        # The bundled models were serialized with different scikit-learn versions.
        # Suppress only the pickle-version warning here; this does NOT make the
        # serialized estimators version-compatible or validate their predictions.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InconsistentVersionWarning)
            model_object = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model {model_path}. Check the training-time and runtime "
            f"scikit-learn/NumPy/joblib compatibility. Original error: {exc}"
        ) from exc

    if isinstance(model_object, dict):
        if "model" not in model_object:
            raise ValueError(
                f"Model bundle {model_path} has no 'model' entry. "
                f"Available keys: {list(model_object.keys())}"
            )
        if not hasattr(model_object["model"], "predict"):
            raise ValueError(f"The estimator stored in {model_path} does not expose predict().")
        return model_object

    if not hasattr(model_object, "predict"):
        raise ValueError(f"Loaded object from {model_path} cannot make predictions.")
    return model_object


def unwrap_model(model_or_bundle):
    return model_or_bundle["model"] if isinstance(model_or_bundle, dict) else model_or_bundle


def _bundle_features(model_or_bundle) -> Optional[list[str]]:
    if not isinstance(model_or_bundle, dict):
        return None
    features = model_or_bundle.get("features")
    if features is None:
        return None
    names = [str(name) for name in list(features)]
    if not names:
        return None
    if len(names) != len(set(names)):
        raise ValueError("The model bundle contains duplicate feature names.")
    return names


def predict_matrix(model, X: pd.DataFrame):
    """Run model inference and return prediction plus an auxiliary score."""
    prediction = np.asarray(model.predict(X))
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        score = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] >= 2 else prediction
    elif hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X))
    else:
        score = prediction
    return prediction, score


def align_features_to_model(model, X: pd.DataFrame, mode_label: str) -> pd.DataFrame:
    """Reorder columns to the estimator's fitted feature_names_in_, if present."""
    fitted_names = getattr(model, "feature_names_in_", None)
    if fitted_names is None:
        return X
    fitted_names = [str(name) for name in fitted_names]
    missing = [name for name in fitted_names if name not in X.columns]
    if missing:
        raise ValueError(
            f"{mode_label} feature mismatch: model expects {len(fitted_names)} named features, "
            f"but {len(missing)} are missing. Examples: {missing[:20]}"
        )
    return X.loc[:, fitted_names].copy()


def _apply_bundle_feature_order(model_or_bundle, estimator, X: pd.DataFrame, mode_label: str) -> pd.DataFrame:
    bundled_features = _bundle_features(model_or_bundle)
    if bundled_features is not None:
        missing = [name for name in bundled_features if name not in X.columns]
        if missing:
            raise ValueError(
                f"{mode_label} model bundle expects {len(bundled_features)} features, "
                f"but {len(missing)} are missing. Examples: {missing[:20]}"
            )
        X = X.loc[:, bundled_features].copy()
    return align_features_to_model(estimator, X, mode_label)


def _inverse_target_transform(raw_prediction: np.ndarray, mode: str) -> np.ndarray:
    """Preserve the inverse target transforms used by the supplied models."""
    raw_prediction = np.asarray(raw_prediction, dtype=float).reshape(-1)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        if mode == "smiles":
            prediction = np.power(10.0, -raw_prediction)
        elif mode in {"natural", "modified", "structure"}:
            prediction = np.power(10.0, raw_prediction)
        else:
            raise ValueError(f"Unsupported target transformation mode: {mode}")
    if not np.all(np.isfinite(prediction)):
        raise ValueError(
            f"Inverse target transformation for {mode} produced non-finite values. "
            "Check the model output and its training-time target transformation."
        )
    return prediction


# ---------------------------------------------------------------------------
# Sequence prediction
# ---------------------------------------------------------------------------
def _read_sequence_inputs(input_paths: Sequence[Path | str]) -> list[tuple[str, str]]:
    """Read and combine any number of FASTA/plain-sequence input files."""
    records: list[tuple[str, str]] = []
    for input_path in input_paths:
        records.extend(read_sequence_input(input_path))
    if not records:
        raise ValueError("No peptide sequences were found in the supplied input files.")
    unique_ids = _make_unique_ids([record[0] for record in records])
    return [(unique_ids[index], record[1]) for index, record in enumerate(records)]


def sequence_mode(args, workdir: Path) -> pd.DataFrame:
    records = _read_sequence_inputs(args.input)
    ids = [record[0] for record in records]
    sequences = [record[1] for record in records]

    workdir.mkdir(parents=True, exist_ok=True)
    fasta_path = workdir / "sequences_for_pfeature.fasta"
    with fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for seq_id, sequence in records:
            handle.write(f">{seq_id}\n{sequence}\n")

    model_object = load_model(DEFAULT_SEQUENCE_MODEL)
    model = unwrap_model(model_object)
    selected_descriptors = read_feature_names(DEFAULT_SEQUENCE_FEATURES)
    if len(selected_descriptors) != 100:
        raise ValueError(
            f"Expected exactly 100 selected sequence descriptors in "
            f"{DEFAULT_SEQUENCE_FEATURES}, found {len(selected_descriptors)}."
        )

    feature_csv = workdir / "pfeature_sequence_model.csv"
    pfeature_df = run_pfeature(
        fasta_path,
        feature_csv,
        SEQUENCE_FEATURE_TYPES,
        DEFAULT_PFEATURE,
        sys.executable,
    )
    if len(pfeature_df) != len(records):
        raise ValueError(
            f"Pfeature returned {len(pfeature_df)} rows for {len(records)} input sequences. "
            "Prediction stopped to prevent mismatching IDs and descriptors."
        )

    missing = [column for column in selected_descriptors if column not in pfeature_df.columns]
    if missing:
        raise ValueError(
            "Pfeature output is missing descriptors required by the sequence model: "
            + ", ".join(missing[:25])
        )

    X = pfeature_df.loc[:, selected_descriptors].copy()
    modifications = parse_modifications(args.modifications) if args.mode == "modified" else []
    for column in MOD_FLAG_COLUMNS:
        X[column] = 1 if column in modifications else 0
    X = X.loc[:, selected_descriptors + MOD_FLAG_COLUMNS].copy()
    X = _apply_bundle_feature_order(model_object, model, X, f"{args.mode} sequence")

    try:
        raw_prediction, _ = predict_matrix(model, X)
    except Exception as exc:
        raise RuntimeError(
            f"Prediction failed for {args.mode} sequence mode. "
            f"Feature matrix shape={X.shape}. Original error: {exc}"
        ) from exc

    prediction = _inverse_target_transform(raw_prediction, args.mode)
    if len(prediction) != len(records):
        raise RuntimeError(
            f"The sequence model returned {len(prediction)} predictions for {len(records)} sequences."
        )

    result = pd.DataFrame({
        "ID": ids,
        "Sequence": sequences,
        "Prediction": prediction,
        "InputType": args.mode,
        "Modifications": ",".join(modifications) if modifications else "none",
    })
    result["ModificationFlags"] = [
        ";".join(f"{column}={int(X.iloc[row_index][column])}" for column in MOD_FLAG_COLUMNS)
        for row_index in range(len(X))
    ]
    return result


# ---------------------------------------------------------------------------
# Chemical input and Mordred descriptor generation
# ---------------------------------------------------------------------------
def read_smiles_input(path: Path | str) -> list[tuple[str, str]]:
    """Read one SMILES per line; an optional second token is used as its ID."""
    input_path = require_file(path, "SMILES input")
    lines = [line.strip() for line in input_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError("SMILES input file is empty.")

    records: list[tuple[str, str]] = []
    for index, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) > 2:
            raise ValueError(
                f"SMILES line {index} contains more than two whitespace-separated fields. "
                "Use: SMILES [optional_ID]."
            )
        smiles = parts[0]
        molecule_id = parts[1] if len(parts) == 2 else f"Molecule_{index}"
        records.append((molecule_id, smiles))

    unique_ids = _make_unique_ids([record[0] for record in records])
    return [(unique_ids[i], records[i][1]) for i in range(len(records))]


def _read_smiles_inputs(input_paths: Sequence[Path | str]) -> list[tuple[str, str]]:
    """Read and combine SMILES entries from any number of input files."""
    records: list[tuple[str, str]] = []
    for input_path in input_paths:
        records.extend(read_smiles_input(input_path))
    if not records:
        raise ValueError("No SMILES records were found in the supplied input files.")
    unique_ids = _make_unique_ids([record[0] for record in records])
    return [(unique_ids[index], record[1]) for index, record in enumerate(records)]


def _import_local_mordred():
    """Import the exact vendored Mordred-community tree used by model development."""
    mordred_package = DEFAULT_MORDRED_DIR / "mordred" / "__init__.py"
    if not mordred_package.is_file():
        raise FileNotFoundError(
            "The vendored Mordred-community source was not found. Expected: "
            f"{mordred_package}. Keep the models/mordred-community directory unchanged."
        )

    local_path = str(DEFAULT_MORDRED_DIR.resolve())
    if local_path in sys.path:
        sys.path.remove(local_path)
    sys.path.insert(0, local_path)

    try:
        import mordred
        from mordred import Calculator, descriptors
    except Exception as exc:
        raise ImportError(
            f"Could not import the vendored Mordred-community source at {local_path}: {exc}"
        ) from exc

    imported_file = Path(mordred.__file__).resolve()
    try:
        imported_file.relative_to(DEFAULT_MORDRED_DIR.resolve())
    except ValueError as exc:
        raise ImportError(
            "A different Mordred installation was imported instead of the vendored "
            f"source. Imported: {imported_file}; expected under: {DEFAULT_MORDRED_DIR}"
        ) from exc

    return Calculator, descriptors, mordred


def smiles_to_mordred(smiles_records: Sequence[tuple[str, str]]):
    """Convert SMILES to RDKit molecules and calculate the vendored Mordred 2D descriptors."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("SMILES and structure modes require RDKit in the active Python environment.") from exc

    Calculator, descriptors, mordred = _import_local_mordred()
    log(f"Using vendored Mordred source: {mordred.__file__}")

    molecules = []
    names = []
    canonical_smiles = []
    for name, smiles in smiles_records:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES for '{name}': {smiles}")
        Chem.SanitizeMol(molecule)
        molecules.append(molecule)
        names.append(name)
        canonical_smiles.append(Chem.MolToSmiles(molecule, isomericSmiles=True))

    descriptor_modules = descriptors.all
    if not descriptor_modules:
        raise RuntimeError("Vendored Mordred imported, but descriptors.all is empty.")

    calculator = Calculator(descriptor_modules, ignore_3D=True)
    if len(calculator.descriptors) == 0:
        raise RuntimeError("Mordred initialized with zero descriptors; check the vendored source tree.")

    log(
        f"Calculating Mordred 2D descriptors (ignore_3D=True; "
        f"{len(calculator.descriptors)} descriptors)."
    )
    # Mordred can emit this RuntimeWarning for an eigenvalue-based descriptor
    # when an intermediate value is mathematically undefined. Keep the resulting
    # NaN/inf values intact: downstream mode-specific validation/preprocessing
    # decides whether they can be handled. Suppress only this known warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"invalid value encountered in scalar power",
            category=RuntimeWarning,
            module=r"mordred\._matrix_attributes",
        )
        descriptor_df = calculator.pandas(molecules, nproc=1)
    descriptor_df.columns = [str(column) for column in descriptor_df.columns]
    for column in descriptor_df.columns:
        descriptor_df[column] = pd.to_numeric(descriptor_df[column], errors="coerce")
    return names, canonical_smiles, descriptor_df


def load_selected_features(feature_file: Path | str, descriptor_df: pd.DataFrame) -> pd.DataFrame:
    """Select SMILES model descriptors in the exact feature-list order."""
    selected = read_feature_names(feature_file, expected_column="Feature", delimiter=",")
    missing = [feature for feature in selected if feature not in descriptor_df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} selected feature(s) are absent from Mordred output. "
            f"Examples: {missing[:20]}. Confirm the feature list and vendored Mordred version."
        )

    X = descriptor_df.loc[:, selected].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    bad = X.columns[X.isna().any()].tolist()
    if bad:
        raise ValueError(
            "Mordred generated missing/non-numeric values for selected SMILES features: "
            + ", ".join(bad[:25])
            + ". No imputation is applied here; use the same preprocessing as model training."
        )
    return X


# ---------------------------------------------------------------------------
# PDB structure conversion. This preserves the original RDKit workflow.
# ---------------------------------------------------------------------------
def sanitize_with_valence_fix(molecule, max_attempts: int = 10):
    """Apply the legacy RDKit sanitization/charge-adjustment procedure."""
    from rdkit import Chem

    for _ in range(max_attempts):
        try:
            Chem.SanitizeMol(molecule)
            return molecule
        except Chem.AtomValenceException as exc:
            match = re.search(r"atom # (\d+)", str(exc))
            if not match:
                raise ValueError(f"Could not identify atom with valence error: {exc}") from exc
            atom = molecule.GetAtomWithIdx(int(match.group(1)))
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        except Exception as exc:
            raise ValueError(f"RDKit sanitization failed: {exc}") from exc
    raise ValueError("RDKit sanitization failed after repeated valence-fix attempts.")


def structure_to_smiles(pdb_path: Path | str) -> str:
    """Convert a PDB containing the peptide/ligand of interest to a single SMILES.

    As in the original implementation, the largest connected molecular fragment
    is retained. The input should not be a complete protein-receptor complex.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("Structure mode requires RDKit in the active Python environment.") from exc

    pdb_file = require_file(pdb_path, "PDB input")
    molecule = Chem.MolFromPDBFile(str(pdb_file), removeHs=False, sanitize=False)
    if molecule is None:
        raise ValueError(f"RDKit could not read the PDB file: {pdb_file}")

    molecule = sanitize_with_valence_fix(molecule)
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if not fragments:
        raise ValueError("No molecular fragments were found in the PDB.")
    molecule = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())
    molecule = Chem.RemoveHs(molecule)
    smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
    if not smiles:
        raise ValueError("PDB-to-SMILES conversion returned an empty string.")
    return smiles


# ---------------------------------------------------------------------------
# SMILES/PDB prediction
# ---------------------------------------------------------------------------
def chemical_mode(args, workdir: Path) -> pd.DataFrame:
    if args.mode == "smiles":
        records = _read_smiles_inputs(args.input)
        model_path = DEFAULT_SMILES_MODEL
        feature_path = DEFAULT_SMILES_FEATURES
    elif args.mode == "structure":
        records = []
        for input_path in args.input:
            structure_path = require_file(input_path, "PDB input")
            structure_smiles = structure_to_smiles(structure_path)
            log(f"Structure-derived SMILES for {structure_path.name}: {structure_smiles}")
            records.append((structure_path.stem, structure_smiles))
        structure_ids = _make_unique_ids([record[0] for record in records])
        records = [(structure_ids[index], record[1]) for index, record in enumerate(records)]
        model_path = DEFAULT_STRUCTURE_MODEL
        feature_path = None
    else:
        raise ValueError(f"Unsupported chemical prediction mode: {args.mode}")

    names, canonical, descriptors_df = smiles_to_mordred(records)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smiles":
        full_features = descriptors_df.copy()
        feature_table_path = workdir / "smiles_all_mordred_descriptors.csv"
        X = load_selected_features(feature_path, full_features)
        log(f"SMILES selected feature list: {X.shape[1]} features")
    else:
        full_features = descriptors_df.copy()
        full_features.columns = ["Mordred_" + str(column) for column in full_features.columns]
        feature_table_path = workdir / "structure_all_mordred_descriptors.csv"
        X = full_features.copy()

    saved_features = full_features.copy()
    saved_features.insert(0, "ID", names)
    saved_features.to_csv(feature_table_path, index=False)
    log(f"Full Mordred feature table saved to {feature_table_path}")

    model_object = load_model(model_path)
    model = unwrap_model(model_object)
    X = _apply_bundle_feature_order(model_object, model, X, args.mode)
    X = X.apply(pd.to_numeric, errors="coerce")

    values = X.to_numpy(dtype=float)
    if args.mode == "structure":
        # The structure model was trained and saved as a Pipeline with the
        # original DescriptorPreprocessor in its "preprocess" step. That fitted
        # transformer is responsible for converting +/-inf to NaN, applying
        # training-set medians, and adding the missingness indicators learned
        # during training. Do not reject Mordred NaNs before model.predict().
        preprocess_step = getattr(model, "named_steps", {}).get("preprocess")
        if preprocess_step is None or not hasattr(preprocess_step, "medians_"):
            raise ValueError(
                "The structure model does not expose its fitted 'preprocess' step. "
                "Expected the original DescriptorPreprocessor pipeline so that "
                "missing Mordred descriptors are handled using training-set medians."
            )
        log(
            "Structure descriptors include missing/non-finite values; "
            "passing them to the fitted DescriptorPreprocessor for "
            "training-consistent handling."
        )
    else:
        if np.isinf(values).any():
            bad = X.columns[np.isinf(values).any(axis=0)].tolist()
            raise ValueError(
                f"Infinite values found in {args.mode} model input: {bad[:30]}. "
                "Check Mordred descriptor calculation."
            )
        if np.isnan(values).any():
            bad = X.columns[np.isnan(values).any(axis=0)].tolist()
            raise ValueError(
                f"Missing/non-numeric values found in {args.mode} model input: {bad[:30]}. "
                "No new imputation is introduced; preserve the model's training-time preprocessing."
            )

    log(f"Final feature matrix for {args.mode}: {X.shape[0]} sample(s), {X.shape[1]} feature(s)")
    try:
        raw_prediction, _ = predict_matrix(model, X)
    except Exception as exc:
        raise RuntimeError(
            f"Prediction failed for {args.mode} mode. Feature matrix shape={X.shape}. "
            f"Original error: {exc}"
        ) from exc

    prediction = _inverse_target_transform(raw_prediction, args.mode)
    if len(prediction) != len(names):
        raise RuntimeError(
            f"The {args.mode} model returned {len(prediction)} predictions for {len(names)} inputs."
        )

    return pd.DataFrame({
        "ID": names,
        "InputType": args.mode,
        "SMILES": canonical,
        "Prediction": prediction,
    })



# ---------------------------------------------------------------------------
# Integrated design mode
# ---------------------------------------------------------------------------
def _generate_natural_design(sequence: str, input_id: str) -> list[dict]:
    """Generate WT plus all non-wild-type single-residue substitutions."""
    variants = [{
        "Input_ID": input_id,
        "Variant": f"{input_id}__WT",
        "Mutation": "Wild-type reference",
        "Position": "",
        "Original_AA": "",
        "New_AA": "",
        "Sequence": sequence,
        "ModificationCombination": "00000",
    }]
    for index, original_aa in enumerate(sequence):
        position = index + 1
        for new_aa in sorted(AA20):
            if new_aa == original_aa:
                continue
            mutation = f"{original_aa}{position}{new_aa}"
            mutant = sequence[:index] + new_aa + sequence[index + 1:]
            variants.append({
                "Input_ID": input_id,
                "Variant": f"{input_id}__{mutation}",
                "Mutation": mutation,
                "Position": position,
                "Original_AA": original_aa,
                "New_AA": new_aa,
                "Sequence": mutant,
                "ModificationCombination": "00000",
            })
    return variants


def _validate_design_sequences(records: Sequence[tuple[str, str]]) -> None:
    """Apply the original design-module length range to every designed peptide."""
    invalid = [
        (seq_id, len(sequence))
        for seq_id, sequence in records
        if not DESIGN_MIN_SEQUENCE_LENGTH <= len(sequence) <= DESIGN_MAX_SEQUENCE_LENGTH
    ]
    if invalid:
        details = "; ".join(
            f"{seq_id}: {length} aa" for seq_id, length in invalid[:20]
        )
        raise ValueError(
            f"Design mode supports peptide lengths of {DESIGN_MIN_SEQUENCE_LENGTH}-"
            f"{DESIGN_MAX_SEQUENCE_LENGTH} amino acids. Out-of-range input(s): {details}"
            + (f"; and {len(invalid) - 20} more" if len(invalid) > 20 else "")
        )


def design_mode(args, workdir: Path) -> pd.DataFrame:
    """Run natural mutation scanning or all 32 modification combinations.

    Unlike the original standalone design_predict.py, this integrated version
    accepts multiple FASTA/plain-sequence files and any number of records.
    """
    if args.mode not in {"natural", "modified"}:
        raise ValueError("--design is available only with --mode natural or --mode modified.")

    input_records = _read_sequence_inputs(args.input)
    _validate_design_sequences(input_records)

    selected_descriptors = read_feature_names(DEFAULT_SEQUENCE_FEATURES)
    if len(selected_descriptors) != 100:
        raise ValueError(
            f"Expected exactly 100 selected sequence descriptors in "
            f"{DEFAULT_SEQUENCE_FEATURES}, found {len(selected_descriptors)}."
        )

    # Prepare the Pfeature input. Natural design calculates descriptors for
    # every mutant; modified design calculates one descriptor row per input
    # peptide and expands it over the 32 modification combinations.
    feature_input_records: list[tuple[str, str]] = []
    metadata: list[dict] = []

    if args.mode == "natural":
        for input_id, sequence in input_records:
            variants = _generate_natural_design(sequence, input_id)
            for variant in variants:
                feature_id = f"design_{len(feature_input_records) + 1}"
                feature_input_records.append((feature_id, variant["Sequence"]))
                for flag in MOD_COLUMNS:
                    variant[flag] = 0
                metadata.append(variant)
    else:
        for input_id, sequence in input_records:
            feature_id = f"base_{len(feature_input_records) + 1}"
            feature_input_records.append((feature_id, sequence))

    workdir.mkdir(parents=True, exist_ok=True)
    fasta_path = workdir / "design_sequences_for_pfeature.fasta"
    with fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for feature_id, sequence in feature_input_records:
            handle.write(f">{feature_id}\n{sequence}\n")

    feature_csv = workdir / "design_pfeature_descriptors.csv"
    pfeature_df = run_pfeature(
        fasta_path,
        feature_csv,
        SEQUENCE_FEATURE_TYPES,
        DEFAULT_PFEATURE,
        sys.executable,
    )
    if len(pfeature_df) != len(feature_input_records):
        raise ValueError(
            f"Pfeature returned {len(pfeature_df)} rows for "
            f"{len(feature_input_records)} design input sequence(s). "
            "Design prediction stopped to prevent feature/sequence mismatching."
        )

    missing = [column for column in selected_descriptors if column not in pfeature_df.columns]
    if missing:
        raise ValueError(
            "Pfeature output is missing descriptors required by the sequence model: "
            + ", ".join(missing[:25])
        )
    base_features = pfeature_df.loc[:, selected_descriptors].reset_index(drop=True)

    if args.mode == "natural":
        X = base_features.copy()
        for flag in MOD_COLUMNS:
            X[flag] = 0
        X = X.loc[:, selected_descriptors + MOD_COLUMNS].copy()
    else:
        rows: list[pd.DataFrame] = []
        combinations = DESIGN_FLAG_COMBINATIONS
        for row_index, (input_id, sequence) in enumerate(input_records):
            for combination in combinations:
                row = base_features.iloc[[row_index]].copy()
                for flag_index, flag in enumerate(MOD_COLUMNS):
                    row[flag] = int(combination[flag_index])
                rows.append(row)

                combination_label = "".join(str(value) for value in combination)
                variant = f"{input_id}__MOD_{combination_label}"
                metadata.append({
                    "Input_ID": input_id,
                    "Variant": variant,
                    "Mutation": "No residue substitution",
                    "Position": "",
                    "Original_AA": "",
                    "New_AA": "",
                    "Sequence": sequence,
                    "ModificationCombination": combination_label,
                    **{
                        flag: int(combination[flag_index])
                        for flag_index, flag in enumerate(MOD_COLUMNS)
                    },
                })
        X = pd.concat(rows, ignore_index=True)
        X = X.loc[:, selected_descriptors + MOD_COLUMNS].copy()

    model_object = load_model(DEFAULT_SEQUENCE_MODEL)
    model = unwrap_model(model_object)
    X = _apply_bundle_feature_order(model_object, model, X, f"design {args.mode}")

    try:
        raw_prediction, _ = predict_matrix(model, X)
    except Exception as exc:
        raise RuntimeError(
            f"Design prediction failed for {args.mode} mode. "
            f"Feature matrix shape={X.shape}. Original error: {exc}"
        ) from exc

    prediction = _inverse_target_transform(raw_prediction, args.mode)
    if len(prediction) != len(metadata):
        raise RuntimeError(
            f"The design model returned {len(prediction)} predictions for "
            f"{len(metadata)} generated variants."
        )

    result = pd.DataFrame(metadata)
    result["Prediction"] = prediction
    result["Mode"] = f"design_{args.mode}"

    # Rank variants separately for each submitted peptide, highest predicted
    # half-life first. This remains meaningful when several input sequences
    # are designed in the same run.
    result = result.sort_values(
        ["Input_ID", "Prediction"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    result.insert(0, "Rank", result.groupby("Input_ID").cumcount() + 1)

    ordered_columns = [
        "Rank", "Input_ID", "Variant", "Mutation", "Position",
        "Original_AA", "New_AA", "Sequence", "ModificationCombination",
    ] + MOD_COLUMNS + ["Prediction", "Mode"]
    return result.loc[:, ordered_columns]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plifepred2_master_v4.py",
        description=(
            "PLifePred2 standalone pipeline. Prediction is the default; "
            "use --design for mutation/modification design."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["natural", "modified", "smiles", "structure"],
        help="Input route: natural peptide, modified peptide, SMILES, or PDB structure.",
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help=(
            "One or more input files. FASTA files may contain any number of sequences; "
            "SMILES files may contain any number of lines; structure mode accepts one "
            "or more PDB files."
        ),
    )
    parser.add_argument(
        "--output", "-o", default="plifepred2_predictions.tsv",
        help="Output TSV file.",
    )
    parser.add_argument(
        "--design",
        action="store_true",
        help=(
            "Run design instead of ordinary prediction. Supports natural "
            "single-substitution scanning or all 32 modification combinations."
        ),
    )
    parser.add_argument(
        "--modifications", "--mods", default="",
        help="Prediction mode only: comma-separated d,ct,nt,cyc,ptm (or legacy values 0-4).",
    )
    parser.add_argument(
        "--workdir", default=None,
        help="Optional directory for intermediate descriptor files. A temporary directory is used by default.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.design and args.mode not in {"natural", "modified"}:
        raise SystemExit("[ERROR] --design supports only --mode natural or --mode modified.")
    if args.design and args.mode == "modified" and args.modifications:
        log("--modifications is ignored in modified design; all 32 flag combinations are generated.")
    elif not args.design and args.mode != "modified" and args.modifications:
        log("--modifications is ignored unless --mode modified is selected.")
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_workdir = None
    try:
        if args.workdir:
            workdir = Path(args.workdir).expanduser().resolve()
            workdir.mkdir(parents=True, exist_ok=True)
            keep_workdir = True
        else:
            temporary_workdir = tempfile.TemporaryDirectory(prefix="plifepred2_")
            workdir = Path(temporary_workdir.name)
            keep_workdir = False

        if args.design:
            result = design_mode(args, workdir)
            operation = "Design"
        elif args.mode in {"natural", "modified"}:
            result = sequence_mode(args, workdir)
            operation = "Prediction"
        else:
            result = chemical_mode(args, workdir)
            operation = "Prediction"

        result.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
        log(f"{operation} completed: {len(result)} result(s) saved to {output_path}")
        if keep_workdir:
            log(f"Intermediate files retained in {workdir}")
        print(result.to_string(index=False))
        return 0

    except KeyboardInterrupt:
        print("[ERROR] Operation interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary_workdir is not None:
            temporary_workdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

