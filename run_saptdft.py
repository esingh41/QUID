"""Run SAPT0/aug-cc-pV(D+d)Z on the Na+...water separation scan.

Loops the rows of ``na_water_dimers.pkl`` (built by build_na_water_systems.py),
runs ``psi4.energy('sapt0')`` on each 2-fragment dimer, and writes the component
energies (kcal/mol) back onto the pickle, mirroring the s66x8_dimer_multipoles.pkl
schema so the existing model/plot code can consume it.

Run in an env with Psi4 (omol25 or qcml_10):
    conda run -n omol25 python run_sapt0.py
"""
import os

import numpy as np
import pandas as pd
import psi4
import qcelemental as qcel
import time

H2KCAL = qcel.constants.conversion_factor("hartree", "kcal/mol")

PICKLE = "na_water_dimers.pkl"
PICKLE = "131_Na-benzene.pkl"
ID_COL = "system_id"

P4_SAPTPBE0_OPTIONS = {
    "basis": "aug-cc-pV(D+d)Z",
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
    # "SCF_SUBTYPE": "OUT_OF_CORE",
}

def _saptdft(qcel_mol, job_id):
    """Run SAPT0/aug-cc-pV(D+d)Z on one dimer; return a dict of kcal/mol energies."""
    psi4.core.clean()
    psi4.geometry(qcel_mol.to_string("psi4", "angstrom"))
    psi4.set_options(P4_SAPTPBE0_OPTIONS)
    psi4.energy("sapt(dft)-d4(i)")
    psi4.core.print_variables()
    # psi4.core.get_timer_records()
    # print(psi4.core.get_timer_records())

    # Convert every variable to units kcalmol here.
    def var(name):
        return psi4.core.variable(name) * H2KCAL

    all_vars = psi4.core.variables()
    for key, val in all_vars.items(): 
        print(key, val)

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
    assert np.allclose(total, total_comp, atol=-1e-8), "Component totals do not sum to SAPT(DFT) total...uh oh"
    print(f"{ind = }")
    print(f"{ind + delta_hf = }")
    print(f"{ind20r + exch_ind20r + delta_hf = }")
    assert np.allclose(ind, ind20r + exch_ind20r + delta_hf, atol=-1e-8), "Induction components to do not sum to sapt(dft) induction...uh_oh"

    elst_hf = var("SAPT(HF) ELST ENERGY")
    exch_hf = var("SAPT(HF) EXCH ENERGY")
    ind_hf = var("SAPT(HF) IND ENERGY")
    disp_hf = var("SAPT DISP ENERGY")
    total_hf = elst_hf + exch_hf + ind_hf + disp_hf

    ind20r_hf = var("SAPT(HF) IND20,R")
    exch_ind20r_hf = var("SAPT(HF) EXCH-IND20,R")


    #Important so if source code ever changes, and delta HF somehow magically includes it we will know
    assert np.allclose(ind_hf, ind20r_hf + exch_ind20r_hf, atol=-1e-8), "Induction components to do not sum to sapt(hf) WIHOUT delta-HF induction...uh_oh"
    return {
        "SAPT(PBE0)-D4(i) ELST kcalmol": elst,
        "SAPT(PBE0)-D4(i) EXCH kcalmol": exch,
        "SAPT(PBE0)-D4(i) INDU kcalmol": ind,
        "SAPT(PBE0)-D4(i) DISP kcalmol": disp,
        "SAPT(PBE0)-D4(i) TOTAL kcalmol": total,
        # split induction, straight from this SAPT0 job (no external reference pkl)
        "SAPT(PBE0)-D4(i) ind20,r kcalmol": ind20r,
        "SAPT(PBE0)-D4(i) exch-ind20,r kcalmol": exch_ind20r,
        "delta HF correction": delta_hf,
        "SAPT0-D4(i) ELST kcalmol" : elst_hf, 
        "SAPT0-D4(i) EXCH kcalmol" : exch_hf, 
        "SAPT0-D4(i) IND  kcalmol" : ind_hf + delta_hf, # Must add delta HF correction to get the true total induction energy 
        "SAPT0-D4(i) DISP kcalmol" : disp, 
        "SAPT0-D4(i) ind20,r kcalmol": ind20r_hf,
        "SAPT0-D4(i) exch-ind20,r kcalmol": exch_ind20r_hf,
        "SAPT0-D4(i) TOTAL kcalmol" : elst_hf + exch_hf + ind_hf + disp, 
    }


def run_saptdft(filename=PICKLE):
    psi4.set_memory("32 GB")
    psi4.set_num_threads(10)
    os.makedirs("saptdft_output", exist_ok=True)

    df = pd.read_pickle(filename)
    cols = {}
    for n, r in df.iterrows():
        job_id = r[ID_COL]
        psi4.core.set_output_file(f"saptdft_output/saptdft_{job_id}.out", True)


        start = time.perf_counter()
        res = _saptdft(r["qcel_molecule"], job_id)
        end = time.perf_counter()
        elapsed = start - end
        print(f"Wall time: {elapsed} seconds")
        for k, v in res.items():
            cols.setdefault(k, []).append(v)
        cols.setdefault("wall time", []).append(elapsed)
    for col, vals in cols.items():
        df[col] = vals
    df.to_pickle(filename)
    print(f"\nWrote SAPT(DFT) and SAPT(HF) columns to {filename}")
    return df


if __name__ == "__main__":
    run_saptdft()
