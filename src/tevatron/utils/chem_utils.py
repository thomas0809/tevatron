import numpy as np
# import scipy
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
# from scipy import sparse


def get_mol_fp(mol_smi: str, fp_radius: int = 2, fp_size: int = 2048, dtype: str = "int32") -> np.ndarray:
    mol_smi = "".join(mol_smi.split())
    fp_generator = GetMorganGenerator(
        radius=fp_radius, countSimulation=True, includeChirality=True, fpSize=fp_size
    )
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_generator.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    # sparse_fp = sparse.csr_matrix(count_fp, dtype=dtype)

    return count_fp


def get_rxn_fp(rxn_smi: str, **kwargs) -> np.ndarray:
    reactant, product = rxn_smi.split(">>")
    kwargs['fp_size'] //= 2
    reactant_fp = get_mol_fp(reactant, **kwargs)
    product_fp = get_mol_fp(product, **kwargs)
    # return product_fp - reactant_fp
    return np.concatenate((reactant_fp, product_fp))


def get_fingerprint(smiles: str, **kwargs) -> np.ndarray:
    if ">>" not in smiles:
        return get_mol_fp(smiles, **kwargs)
    else:
        return get_rxn_fp(smiles, **kwargs)
