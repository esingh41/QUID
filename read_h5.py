import h5py
import qcelemental as qcel
import numpy as np
import pandas as pd

ANGSTROM_TO_BOHR = qcel.constants.conversion_factor("angstrom", "bohr")


def build_qcel_mols(props):
    monA = qcel.models.Molecule(
            symbols=props['atoms']['small_monomer'][:],
            geometry=props['positions']['small_monomer'][:] * ANGSTROM_TO_BOHR,
            fix_com=True,
            fix_orientation=True,
    )
    monB = qcel.models.Molecule(
            symbols=props['atoms']['big_monomer'][:],
            geometry=props['positions']['big_monomer'][:] * ANGSTROM_TO_BOHR,
            fix_com=True,
            fix_orientation=True,
    )

    dimer = qcel.models.Molecule(
            symbols=np.concatenate([monA.symbols, monB.symbols]),
            geometry=np.concatenate([monA.geometry, monB.geometry]),
            fragments=[
                np.arange(0, len(monA.symbols)),
                np.arange(len(monA.symbols), len(monA.symbols) + len(monB.symbols)),
            ],
            fix_com=True,
            fix_orientation=True,
    )
    return monA, monB, dimer

def find_reference_id(props, dissociation_group):
    ref_small = props['positions']['small_monomer'][:]
    ref_big = props['positions']['big_monomer'][:]
    for key, traj_props in dissociation_group.items():
        if (np.allclose(traj_props['positions']['small_monomer'][:], ref_small)
                and np.allclose(traj_props['positions']['big_monomer'][:], ref_big)):
            return key
    return None

def build_row(dimer_id, props, extra_props=None):
    ev2kcalmol = qcel.constants.conversion_factor("eV", "kcal / mol")
    row = {}
    mols = build_qcel_mols(props)
    row['system id'] = dimer_id
    row['qcel molecule A'] = mols[0]
    row['qcel molecule B'] = mols[1]
    row['qcel molecule'] = mols[-1]

    sapt_source = props if 'SAPT' in props.keys() else extra_props
    if sapt_source is not None and 'SAPT' in sapt_source.keys():
        for component, value in sapt_source['SAPT'].items():
            row[f'sSAPT0/jaDZ {component} kcalmol'] = value[()]

    for method, eint in props['Eint'].items():
        row[f'{method} interaction E kcalmol'] = eint[()] * ev2kcalmol
    if extra_props is not None:
        for method, eint in extra_props['Eint'].items():
            col = f'{method} interaction E kcalmol'
            if col not in row:
                row[col] = eint[()] * ev2kcalmol
    return row

def gather_data(h5_path='QUID.h5', out_path='quid_dimers.pkl'):
    # h5py files are python dictionaries
    f = h5py.File(h5_path, 'r')

    data = []
    for dimer_id, props in f.items():
        if 'dissociation' not in props:
            data.append(build_row(dimer_id, props))
        else:
            ref_id = find_reference_id(props, props['dissociation'])
            for key, trajectory_props in props['dissociation'].items():
                extra = props if key == ref_id else None
                data.append(build_row(key, trajectory_props, extra_props=extra))

    df = pd.DataFrame(data)
    # pickle, not parquet/csv: the rows hold live qcelemental Molecule objects
    if out_path is not None:
        df.to_pickle(out_path)
    return df


if __name__ == '__main__':
    df = gather_data()
    print(df.info())
