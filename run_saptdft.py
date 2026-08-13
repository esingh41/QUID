"""Run SAPT(DFT)-D4(i)/aug-cc-pV(D+d)Z over the QUID dimers.

Loops the rows of ``quid_dimers.pkl`` (built by read_h5.py), runs
``psi4.energy('sapt(dft)-d4(i)')`` on each 2-fragment dimer, and records both the
SAPT(PBE0) components and the SAPT0-level components that the delta-HF segment
produces for free.

Each system is ~45 min, so results are checkpointed one file per system into
``saptdft_results/`` the moment that system finishes, then merged into a single
frame at the end. A result file on disk means that system is done: a relaunch
skips it. To force one to rerun, delete its file.

Requires the custom local psi4 build (stock conda psi4 lacks the SAPT_DFT_*
keywords used below). Launch as:
    source worker_ds.sh && python run_saptdft.py
"""
import glob
import os
import time
import traceback

import numpy as np
import pandas as pd
import psi4
import qcelemental as qcel

H2KCAL = qcel.constants.conversion_factor("hartree", "kcal/mol")

IN_PICKLE = "quid_dimers.pkl"
RESULTS_DIR = "saptdft_results"   # one <system id>.pkl per finished system
OUT_PICKLE = "quid_saptdft.pkl"   # the merged frame
OUTDIR = "saptdft_output"         # psi4 .out files

ID_COL = "system id"
MOL_COL = "qcel molecule"
TOTAL_COL = "SAPT(PBE0)-D4(i) TOTAL kcalmol"
SEC_COL = "saptdft seconds"
ERR_COL = "saptdft error"

P4_SAPTPBE0_OPTIONS = {
    "basis": "aug-cc-pV(D+d)Z",
    "scf_type": "df",
    "mp2_type": "df",
    "guess": "SAD",
    "freeze_core": "true",
    "SAPT_DFT_FUNCTIONAL": "PBE0",
    "SAPT_DFT_DO_FSAPT": "NONE", #Setting to NONE because this is expensive in memory
    "SAPT_DFT_USE_EINSUMS": True,
    "FISAPT_FSAPT_FILEPATH": "none",
    "SAPT_DFT_GRAC_COMPUTE": "ITERATIVE",
    "MAXITER": 500,
    # "SCF_SUBTYPE": "OUT_OF_CORE",
}

def _saptdft(qcel_mol):
    """Run SAPT(DFT)-D4(i)/aug-cc-pV(D+d)Z on one dimer; return kcal/mol energies.

    Raises on any psi4 failure, and asserts on any internal inconsistency, so the
    caller can record the system as failed rather than banking suspect numbers.
    """
    psi4.core.clean()
    psi4.core.clean_variables()
    psi4.geometry(qcel_mol.to_string("psi4", "angstrom"))
    psi4.set_options(P4_SAPTPBE0_OPTIONS)
    psi4.energy("sapt(dft)-d4(i)")
    psi4.core.print_variables()

    # Convert every variable to units kcalmol here.
    def var(name):
        return psi4.core.variable(name) * H2KCAL

    # Getting all the SAPT(DFT) energies which are stored plainly
    elst = var("SAPT ELST ENERGY")
    exch = var("SAPT EXCH ENERGY")
    ind = var("SAPT IND ENERGY")
    # This ind ALREADY contains delta HF correction because
    # if "Delta HF Correction" in list(data): in sapt_util.py
    #     ind += data["Delta HF Correction"]

    disp = var("SAPT DISP ENERGY")
    total = var("SAPT TOTAL ENERGY")
    ind20r = var("IND20,R")
    exch_ind20r = var("EXCH-IND20,R")
    delta_hf = var("DELTA HF CORRECTION")
    
    total_comp = elst + exch + ind + disp
    print(f"{total_comp = }")
    print(f"{total = }")
    assert np.allclose(total, total_comp, atol=1e-8), "Component totals do not sum to SAPT(DFT) total...uh oh"
    print(f"{ind = }")
    print(f"{ind + delta_hf = }")
    print(f"{ind20r + exch_ind20r + delta_hf = }")
    assert np.allclose(ind, ind20r + exch_ind20r + delta_hf, atol=1e-8), "Induction components to do not sum to sapt(dft) induction...uh_oh"

    elst_hf = var("SAPT(HF) ELST ENERGY")
    exch_hf = var("SAPT(HF) EXCH ENERGY")
    ind_hf = var("SAPT(HF) IND ENERGY")
    # No HF-level dispersion exists in a SAPT(DFT) run: print_sapt_hf_summary
    # returns before setting SAPT DISP ENERGY whenever delta_hf is passed, so the
    # empirical D4 term above is shared by both totals.

    ind20r_hf = var("SAPT(HF) IND20,R")
    exch_ind20r_hf = var("SAPT(HF) EXCH-IND20,R")


    #Important so if source code ever changes, and delta HF somehow magically includes it we will know
    assert np.allclose(ind_hf, ind20r_hf + exch_ind20r_hf, atol=1e-8), "Induction components to do not sum to sapt(hf) WIHOUT delta-HF induction...uh_oh"
    return {
        "SAPT(PBE0)-D4(i) ELST kcalmol": elst,
        "SAPT(PBE0)-D4(i) EXCH kcalmol": exch,
        "SAPT(PBE0)-D4(i) INDU kcalmol": ind,
        "SAPT(PBE0)-D4(i) DISP kcalmol": disp,
        TOTAL_COL: total,
        # split induction, straight from this SAPT0 job (no external reference pkl)
        "SAPT(PBE0)-D4(i) ind20,r kcalmol": ind20r,
        "SAPT(PBE0)-D4(i) exch-ind20,r kcalmol": exch_ind20r,
        "delta HF correction": delta_hf,
        "SAPT0-D4(i) ELST kcalmol" : elst_hf, 
        "SAPT0-D4(i) EXCH kcalmol" : exch_hf, 
        "SAPT0-D4(i) INDU kcalmol" : ind_hf + delta_hf, # Must add delta HF correction to get the true total induction energy 
        "SAPT0-D4(i) DISP kcalmol" : disp, 
        "SAPT0-D4(i) ind20,r kcalmol": ind20r_hf,
        "SAPT0-D4(i) exch-ind20,r kcalmol": exch_ind20r_hf,
        "SAPT0-D4(i) TOTAL kcalmol" : elst_hf + exch_hf + (ind_hf + delta_hf) + disp, 
    }


def run_one(qcel_mol, job_id):
    """Run one dimer. Returns a record dict; on failure it carries the traceback
    and no energies, so a single bad system cannot take down the other 169."""
    out_file = os.path.join(OUTDIR, f"saptdft_{job_id}.out")
    psi4.core.set_output_file(out_file, False)

    start = time.perf_counter()
    err = None
    res = {}
    try:
        res = _saptdft(qcel_mol)
    except AssertionError:
        # Component sums disagreed. Keep no energies, but tag it so these are
        # greppable apart from SCF/runtime failures.
        err = "INCONSISTENT\n" + traceback.format_exc()
    except Exception:
        err = traceback.format_exc()
    finally:
        # Drop scratch even on failure; 170 jobs would otherwise fill
        # $PSI_SCRATCH (/scratch/esingh41, per worker_ds.sh).
        psi4.core.clean()

    if err and os.path.exists(out_file):
        # set_output_file truncates, so a later retry would erase the evidence.
        os.replace(out_file, out_file + ".failed")

    return {
        ID_COL: job_id,
        SEC_COL: time.perf_counter() - start,
        ERR_COL: err,
        **res,
    }


def merge_results(in_path=IN_PICKLE, results_dir=RESULTS_DIR, out_path=OUT_PICKLE):
    """Merge the per-system checkpoints onto the input frame.

    Safe to call mid-run from another shell to inspect partial progress: systems
    with no result file yet come back as NaN, which is what how="left" gives us.
    """
    records = [
        pd.read_pickle(p)
        for p in sorted(glob.glob(os.path.join(results_dir, "*.pkl")))
    ]
    if not records:
        raise SystemExit(f"no result files in {results_dir}/")

    df = pd.read_pickle(in_path).merge(pd.DataFrame(records), on=ID_COL, how="left")
    df.to_pickle(out_path)
    n_ok = df[TOTAL_COL].notna().sum()
    n_bad = df[ERR_COL].notna().sum()
    print(f"\n{n_ok}/{len(df)} complete, {n_bad} failed -> {out_path}")
    return df


def run_saptdft(in_path=IN_PICKLE):
    psi4.set_memory("64 GB")
    psi4.set_num_threads(10)  # psi4 scales poorly past ~10 threads
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_pickle(in_path)
    for i, row in df.iterrows():
        job_id = row[ID_COL]
        path = os.path.join(RESULTS_DIR, f"{job_id}.pkl")
        # A result file on disk means this system is done. Delete it to rerun.
        if os.path.exists(path):
            print(f"[{i + 1}/{len(df)}] {job_id}: skip (done)")
            continue

        print(f"[{i + 1}/{len(df)}] {job_id}: running...", flush=True)
        record = run_one(row[MOL_COL], job_id)

        # tmp + replace so an interrupt cannot leave a half-written file behind
        # that later breaks the merge.
        tmp = path + ".tmp"
        pd.to_pickle(record, tmp)
        os.replace(tmp, path)

        mins = record[SEC_COL] / 60
        if record[ERR_COL]:
            print(f"    FAILED after {mins:.1f} min: "
                  f"{record[ERR_COL].strip().splitlines()[-1]}")
        else:
            print(f"    done in {mins:.1f} min, "
                  f"total = {record[TOTAL_COL]:.4f} kcal/mol")

    return merge_results(in_path)


if __name__ == "__main__":
    run_saptdft()
