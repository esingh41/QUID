"""Create and submit the QUID SAPT(DFT)-D4(i) dataset through QCFractal.

Reproduces the computation that ``run_sapt.py`` performs locally -- one
``psi4.energy("sapt(dft)-d4(i)")`` per QUID dimer with the P4_SAPTPBE0 option set --
but submitted through QCFractal as one ``SinglepointDatasetEntry`` per dimer instead
of run in a serial loop.  The local route is impractical: ``F2I1_100`` (54 atoms,
3594 basis functions) took 2573 s on 20 threads with 86 GiB of core, so the full 170
systems are roughly five CPU-days serial.

QUID is a single, non-split dataset -- there is no ``DATASET_MASTER`` / per-category
subset pattern here.

Entries are the DIMERS ONLY.  ``read_h5.py`` builds each dimer by concatenating the
small and big monomers, so fragment 0 and fragment 1 of the dimer are geometrically
identical to the ``qcel molecule A`` / ``qcel molecule B`` columns; separate monomer
entries would be redundant, and SAPT cannot run on a single-fragment molecule anyway.

One deliberate deviation from ``run_sapt.py``: ``SCF_SUBTYPE="OUT_OF_CORE"`` is added.
See SAPT_DFT_SPEC's description for why.

Nothing is submitted or deleted unless the matching --action is passed explicitly.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import qcelemental as qcel
import qcportal
from dotenv import load_dotenv
from qcelemental.molparse.chgmult import validate_and_fill_chgmult
from qcportal.record_models import PriorityEnum
from qcportal.singlepoint import SinglepointDatasetEntry

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

load_dotenv()
QCF_USERNAME = os.getenv("QCF_USERNAME", None)
QCF_PASSWORD = os.getenv("QCF_PASSWORD", None)
QCF_URL = os.getenv("QCF_URL", "http://localhost:7778")

if QCF_USERNAME is None or QCF_PASSWORD is None:
    print("Connecting to QCF without authentication")
    client = qcportal.PortalClient(QCF_URL, verify=False)
else:
    print(f"Connecting to QCF as user: {QCF_USERNAME}")
    client = qcportal.PortalClient(
        QCF_URL,
        verify=False,
        username=QCF_USERNAME,
        password=QCF_PASSWORD,
    )

# ---------------------------------------------------------------------------
# Project / dataset constants
# ---------------------------------------------------------------------------

PROJECT_NAME = "quid"
PROJECT_DESCRIPTION = "QUID molecular dimer benchmark set"

# QUID is a single dataset -- no DATASET_MASTER / per-category subsets.
DATASET_NAME = "QUID"
DATASET_DESCRIPTION = (
    "QUID dimers (42 equilibrium + 128 non-equilibrium ligand-pocket structures, "
    "each a large and a small monomer, optimized at PBE0+MBD): SAPT(DFT)-D4(i)/"
    "aug-cc-pV(D+d)Z interaction energy decomposition. Source dataset QUID.h5, "
    "ChemRxiv 10.26434/chemrxiv-2025-f6615."
)

TAG_FREE = "esingh41_free"
TAG_CPU_SMALL = "esingh41_cpu_small"
TAG_CPU_MEDIUM = "esingh41_cpu_medium"
TAG_CPU_LARGE = "esingh41_cpu_large"
TAG_CPU_XLARGE = "esingh41_cpu_xlarge"

# Deliberately the free tag: nothing expensive starts by accident. These are
# 3594-basis-function SAPT(DFT) jobs (~45 min on 20 threads), so real submissions
# should pass --compute-tag esingh41_cpu_xlarge (or _large).
DEFAULT_COMPUTE_TAG = TAG_FREE
DEFAULT_PRIORITY = PriorityEnum.high

proj = client.add_project(
    name=PROJECT_NAME,
    description=PROJECT_DESCRIPTION,
    tagline=PROJECT_NAME,
    default_compute_tag=DEFAULT_COMPUTE_TAG,
    default_compute_priority=DEFAULT_PRIORITY,
    existing_ok=True,
)

# ---------------------------------------------------------------------------
# Dataframe source
# ---------------------------------------------------------------------------

QUID_PICKLE = Path(__file__).resolve().parent / "quid_dimers.pkl"
ID_COL = "system id"
MOLECULE_COL = "qcel molecule"  # the dimer; "qcel molecule A"/"B" are unused here
LABEL = "dimer"

H2KCAL = qcel.constants.conversion_factor("hartree", "kcal/mol")

# All dataframe columns that carry per-system metadata (the 4 reference sSAPT0
# components and the 32 reference interaction energies) rather than a qcel Molecule.
NON_MOLECULE_COLS = None  # populated by load_quid_dataframe()

# The three fields whose absence from the serialized molecule produces the server's
# "Inconsistent or unspecified chg/mult" 400 on add_entries.
_REQUIRED_MOL_FIELDS = {"fragments", "fragment_charges", "fragment_multiplicities"}


def load_quid_dataframe() -> pd.DataFrame:
    """Load quid_dimers.pkl and sanity-check the assumptions this script relies on.

    These are defensive assertions, not filters -- the pickle has already been
    verified clean (170 rows, unique ids, all neutral closed-shell singlets, every
    dimer two-fragment and frozen with fix_com/fix_orientation). If a future rebuild
    of quid_dimers.pkl violates one of these, fail loudly instead of silently
    submitting a bad entry.
    """
    global NON_MOLECULE_COLS

    df = pd.read_pickle(QUID_PICKLE)
    NON_MOLECULE_COLS = [c for c in df.columns if not c.startswith("qcel molecule")]

    if df[ID_COL].duplicated().any():
        dupes = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Duplicate {ID_COL!r} values in {QUID_PICKLE}: {dupes}")

    for _, row in df.iterrows():
        job_id = row[ID_COL]
        mol = row[MOLECULE_COL]

        if len(mol.fragments) != 2:
            raise ValueError(
                f"{ID_COL}={job_id} has {len(mol.fragments)} fragments; expected "
                "exactly 2 (small monomer / big monomer). SAPT needs a dimer."
            )
        if mol.molecular_multiplicity != 1:
            raise ValueError(
                f"{ID_COL}={job_id} is not a closed-shell singlet "
                f"(multiplicity={mol.molecular_multiplicity})"
            )
        if not (mol.fix_com and mol.fix_orientation):
            raise ValueError(
                f"{ID_COL}={job_id} does not have fix_com/fix_orientation set; the "
                "server would re-orient it away from the QUID reference frame"
            )

        # The guard against the chg/mult 400. qcelemental *computes* fragments,
        # fragment_charges and fragment_multiplicities when they are not passed, but
        # they never enter __fields_set__, so qcportal omits them from the payload
        # and the server validator sees [None]. read_h5.py passes fragments=
        # explicitly on the dimer, which pulls all three in -- but its monomers get
        # none of them, so this must be checked rather than assumed.
        serialized = set(json.loads(mol.json()))
        missing = _REQUIRED_MOL_FIELDS - serialized
        if missing:
            raise ValueError(
                f"{ID_COL}={job_id}: molecule does not serialize {sorted(missing)}; "
                "add_entries would fail with 'Inconsistent or unspecified chg/mult'. "
                "Fix this in read_h5.py by passing fragments, fragment_charges and "
                "fragment_multiplicities explicitly, then rebuild quid_dimers.pkl."
            )

        # The server's own validator: catches chemically wrong fragment splits (an
        # odd electron count at multiplicity 1) that the field check above cannot.
        validate_and_fill_chgmult(
            zeff=[int(z) for z in mol.atomic_numbers],
            fragment_separators=[int(f[0]) for f in mol.fragments[1:]],
            molecular_charge=mol.molecular_charge,
            fragment_charges=list(mol.fragment_charges),
            molecular_multiplicity=mol.molecular_multiplicity,
            fragment_multiplicities=list(mol.fragment_multiplicities),
            zero_ghost_fragments=False,
            verbose=0,
        )

    return df


# ---------------------------------------------------------------------------
# QC specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecData:
    name: str
    spec: qcportal.singlepoint.QCSpecification
    description: str


# run_sapt.py's P4_SAPTPBE0_OPTIONS, plus SCF_SUBTYPE (see the spec description).
# F-SAPT is switched on by SAPT_DFT_DO_FSAPT, not by the method string.
SAPT_DFT_PBE0_OPTIONS = {
    "scf_type": "df",
    "mp2_type": "df",
    "guess": "SAD",
    "freeze_core": "true",
    "SAPT_DFT_FUNCTIONAL": "PBE0",
    "SAPT_DFT_DO_FSAPT": "FISAPT",
    "SAPT_DFT_USE_EINSUMS": True,
    "FISAPT_FSAPT_FILEPATH": "none",
    "SAPT_DFT_GRAC_COMPUTE": "ITERATIVE",
    "MAXITER": 500,
    "basis": "aug-cc-pV(D+d)Z",
    # NOT in run_sapt.py -- added because these systems are large. See below.
    "SCF_SUBTYPE": "OUT_OF_CORE",
    # Deliberately NOT set, unlike the omol25 P4_SAPTPBE0 option set:
    #   DF_BASIS_SCF / DF_BASIS_MP2 -- psi4 auto-resolves both -JKFIT and -RI for
    #     every element QUID contains (H, C, N, O, F, P, S, Cl). The
    #     aug-cc-pv_dpd_z-kca overrides there exist only because auxiliary sets were
    #     missing for K/Ca, which QUID lacks.
}

SAPT_DFT_SPEC = SpecData(
    name="sapt(dft)-d4(i)/aug-cc-pV(D+d)Z",
    description=(
        "SAPT(DFT) with the PBE0 functional and D4(i) dispersion in aug-cc-pV(D+d)Z, "
        "with F-SAPT partitioning on."
    ),
    spec=qcportal.singlepoint.QCSpecification(
        program="psi4",
        driver="energy",
        method="sapt(dft)-d4(i)",
        # Must name the same basis set as keywords["basis"]: psi4 clobbers BASIS
        # with the keyword value after keywords are applied
        # (schema_wrapper.py::run_json_qcschema), so a mismatch silently runs a
        # different basis than the name advertises. Case is not part of that --
        # qcportal normalizes this field to "aug-cc-pv(d+d)z" and psi4 resolves
        # basis names case-insensitively.
        basis="aug-cc-pV(D+d)Z",
        keywords=SAPT_DFT_PBE0_OPTIONS,
        protocols=qcportal.singlepoint.SinglepointProtocols(stdout=True),
    ),
)

SPECS = [SAPT_DFT_SPEC]

# ---------------------------------------------------------------------------
# Dataframe row -> entry mapping
# ---------------------------------------------------------------------------


def _json_scalar(value):
    if hasattr(value, "item"):  # numpy scalars are not JSON-serializable
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        # 128 of the 170 rows are non-equilibrium points with no sSAPT0 / high-level
        # reference values, and DMC covers only 13. Bare NaN is not valid JSON.
        return None
    return value


def _entry_attributes(row) -> dict:
    attrs = {col: row[col] for col in NON_MOLECULE_COLS if col != ID_COL}
    attrs["system_id"] = row[ID_COL]
    attrs["label"] = LABEL  # always "dimer"; leaves room for A/B entries later
    return {key: _json_scalar(value) for key, value in attrs.items()}


def build_entries(df: pd.DataFrame) -> list[SinglepointDatasetEntry]:
    return [
        SinglepointDatasetEntry(
            name=row[ID_COL],
            molecule=row[MOLECULE_COL],
            attributes=_entry_attributes(row),
        )
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Idempotent project/dataset setup
# ---------------------------------------------------------------------------


def get_or_create_dataset(name: str = DATASET_NAME, description: str = DATASET_DESCRIPTION):
    client_datasets = [i["dataset_name"] for i in client.list_datasets()]
    if name not in client_datasets:
        ds = proj.add_dataset(
            dataset_type="singlepoint",
            name=name,
            description=description,
            tagline=name,
            default_compute_tag=DEFAULT_COMPUTE_TAG,
            default_compute_priority=DEFAULT_PRIORITY,
            existing_ok=True,
        )
        print(f"Created dataset {name!r}")
    else:
        ds = client.get_dataset("singlepoint", name)
        print(f"Found {name!r} dataset, using this instead")
    print(f"Dataset ID: {ds.id}")
    return ds


def add_missing_entries(ds, entries: list[SinglepointDatasetEntry]) -> None:
    existing = {entry.name for entry in ds.iterate_entries()}
    new_entries = [entry for entry in entries if entry.name not in existing]
    if not new_entries:
        print(f"  {ds.name}: no new entries to add")
        return
    ds.add_entries(new_entries)
    print(f"  {ds.name}: added {len(new_entries)} entries")


def add_specifications(ds, specs: list[SpecData] | None = None) -> None:
    if specs is None:
        specs = SPECS
    for spec in specs:
        try:
            ds.add_specification(
                name=spec.name,
                specification=spec.spec,
                description=spec.description,
            )
            print(f"  {ds.name}: added specification {spec.name}")
        except Exception as exc:
            if "already exists" in str(exc).lower():
                print(f"  {ds.name}: specification {spec.name} already present")
                continue
            raise


# ---------------------------------------------------------------------------
# Init / submit / status
# ---------------------------------------------------------------------------


def init_quid_dataset() -> None:
    """Build (or reuse) the dataset and add any missing entries.

    Does NOT submit any calculations -- the user reviews the dataset before calling
    submit_quid().
    """
    df = load_quid_dataframe()
    ds = get_or_create_dataset()
    add_missing_entries(ds, build_entries(df))
    print(f"Done initializing {DATASET_NAME!r} ({len(df)} dimer entries expected).")


def submit_quid(
    compute_tag: str = DEFAULT_COMPUTE_TAG,
    priority: PriorityEnum = DEFAULT_PRIORITY,
    entry_names: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Submit SAPT_DFT_SPEC against the dataset's entries.

    Every entry is a two-fragment dimer, so entry_names=None ("all 170") is safe --
    unlike the NENCI dataset, there are no monomer entries to keep SAPT away from.
    Use limit to run a pilot batch before releasing the full set.
    """
    ds = client.get_dataset("singlepoint", DATASET_NAME)
    print(f"Processing {ds.name}...")
    add_specifications(ds)

    names = entry_names if entry_names is None else list(entry_names)
    if limit is not None:
        if names is None:
            names = sorted(entry.name for entry in ds.iterate_entries())
        names = names[:limit]

    print(
        f"Submitting {'all' if names is None else len(names)} entries to "
        f"{SAPT_DFT_SPEC.name}"
    )
    ds.submit(
        entry_names=names,
        specification_names=[SAPT_DFT_SPEC.name],
        compute_tag=compute_tag,
        compute_priority=priority,
    )
    print(ds.status_table())


def print_status() -> None:
    ds = client.get_dataset("singlepoint", DATASET_NAME)
    print(f"Dataset: {ds.name}")
    print(ds.status_table())


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

# The four SAPT(DFT) components plus the total, as psi4 qcvars. For a sapt(dft) run
# these are the PBE0-level values. Mirrors run_sapt.py's SAPT_QCVARS.
SAPT_QCVARS = {
    "ELST": "SAPT ELST ENERGY",
    "EXCH": "SAPT EXCH ENERGY",
    "IND": "SAPT IND ENERGY",
    "DISP": "SAPT DISP ENERGY",
    "TOTAL": "SAPT TOTAL ENERGY",
}

# SAPT0-equivalent terms, computed for free inside a SAPT(DFT) run: the delta-HF
# segment (SAPT_DFT_DO_DHF, default true) builds them from HF monomer orbitals, so
# they are true SAPT0 quantities, unlike SAPT_QCVARS above. There is no HF-level
# dispersion counterpart, so a SAPT0 TOTAL cannot be formed from a SAPT(DFT) run;
# that still needs a separate sapt0 specification.
SAPT0_QCVARS = {
    "SAPT0-ELST": "SAPT(HF) ELST ENERGY",
    "SAPT0-EXCH": "SAPT(HF) EXCH ENERGY",
    "SAPT0-IND-NODHF": "SAPT(HF) IND ENERGY",
    "SAPT0-DHF": "SAPT(DFT) Delta HF",
}


def assemble_sapt_record_data(record) -> dict:
    """Pull the SAPT components off one completed SinglepointRecord, in kcal/mol.

    TODO: verify against a real completed record. Which of these land in
    record.properties versus record.extras["qcvars"] is program-specific and not
    derivable from the specification; both are searched here until one run confirms
    it. Run the --action submit --limit 1 pilot first (see the module docstring).
    """
    props = getattr(record, "properties", None) or {}
    extras = getattr(record, "extras", None) or {}
    qcvars = extras.get("qcvars", {}) if isinstance(extras, dict) else {}

    # psi4 upper-cases the keys of variables(), so "SAPT(DFT) Delta HF" is stored as
    # "SAPT(DFT) DELTA HF". Match case-insensitively or the lookup silently yields
    # None (run_sapt.py:99-102).
    merged = {}
    for source in (qcvars, props):
        if isinstance(source, dict):
            merged.update({str(k).upper(): v for k, v in source.items()})

    row = {}
    for short, qcvar in {**SAPT_QCVARS, **SAPT0_QCVARS}.items():
        value = merged.get(qcvar.upper())
        row[f"{short} kcalmol"] = None if value is None else value * H2KCAL

    # SAPT0 induction is conventionally quoted with delta HF folded in.
    ind = merged.get("SAPT(HF) IND ENERGY")
    dhf = merged.get("SAPT(DFT) DELTA HF")
    row["SAPT0-IND kcalmol"] = (
        None if ind is None or dhf is None else (ind + dhf) * H2KCAL
    )
    return row


def quid_sapt_dataframe(
    ds=None, specification_name: str = SAPT_DFT_SPEC.name
) -> pd.DataFrame:
    """One row per system: status plus the SAPT components of completed records."""
    if ds is None:
        ds = client.get_dataset("singlepoint", DATASET_NAME)

    rows: dict[str, dict] = {}
    for entry_name, _, record in ds.iterate_records(
        specification_names=[specification_name]
    ):
        # Read system_id off the attributes rather than parsing entry_name: QUID
        # dissociation ids contain underscores of their own (e.g. "F2B1_090").
        attrs = ds.get_entry(entry_name).attributes or {}
        system_id = attrs.get("system_id")
        if system_id is None:
            continue

        status = getattr(getattr(record, "status", None), "value", str(record.status))
        row = rows.setdefault(system_id, {ID_COL: system_id})
        row["status"] = status
        if status == "complete":
            row.update(assemble_sapt_record_data(record))

    return pd.DataFrame(list(rows.values()))


# ---------------------------------------------------------------------------
# Dataset deletion (destructive -- refuses to act without explicit confirm)
# ---------------------------------------------------------------------------


def delete_dataset(
    name: str = DATASET_NAME,
    delete_records: bool = True,
    confirm: bool = False,
) -> None:
    """Permanently delete a dataset (and, by default, its records) from QCFractal.

    Destructive and hard to reverse -- refuses to act unless confirm=True (wired to
    the --yes CLI flag).
    """
    ds = client.get_dataset("singlepoint", name)
    print(f"About to delete dataset {name!r} (id={ds.id}), delete_records={delete_records}")
    if not confirm:
        print("Refusing to delete without confirmation. Pass confirm=True "
              "(or --yes on the CLI) to proceed.")
        return
    proj.unlink_datasets(
        ds.id,
        delete_dataset_records=delete_records,
        delete_datasets=True,
    )
    print(f"Deleted dataset {name!r} (id={ds.id})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=[
            "init",
            "submit",
            "status",
            "init-submit",
            "delete-dataset",
        ],
        help="Operation to perform.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="With --action submit, submit only the first N entries (sorted by "
             "name). Use for a pilot batch before releasing all 170.",
    )
    parser.add_argument(
        "--compute-tag",
        default=DEFAULT_COMPUTE_TAG,
        help="Compute tag to use for submissions. These are 3594-basis-function "
             f"SAPT(DFT) jobs; prefer {TAG_CPU_XLARGE}.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to actually perform --action delete-dataset; without it, "
             "delete-dataset only prints what it would do.",
    )
    parser.add_argument(
        "--keep-records",
        action="store_true",
        help="With --action delete-dataset, delete the dataset but keep its "
             "underlying records.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help="Dataset name to target, mainly for --action delete-dataset (e.g. to "
             "clean up a dataset created under a since-renamed DATASET_NAME). "
             "Defaults to the current DATASET_NAME.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.action in {"init", "init-submit"}:
        init_quid_dataset()
    if args.action in {"submit", "init-submit"}:
        submit_quid(compute_tag=args.compute_tag, limit=args.limit)
    if args.action == "status":
        print_status()
    if args.action == "delete-dataset":
        delete_dataset(
            name=args.dataset_name,
            delete_records=not args.keep_records,
            confirm=args.yes,
        )


if __name__ == "__main__":
    main()
