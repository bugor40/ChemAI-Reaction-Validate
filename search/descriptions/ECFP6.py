import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


class ECFP6:
    def __init__(self, smiles):
        self.mols = []
        for i in tqdm(smiles):
            try:
                self.mols.append(Chem.MolFromSmiles(i))
            except:
                self.mols.append(None)
        self.smiles = smiles

    def mol2fp(self, mol, radius = 3):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = radius)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def compute_ECFP6(self, name):
        bit_headers = ['bit' + str(i) for i in range(2048)]
        arr = np.empty((0,2048), int).astype(int)
        for i in tqdm(self.mols):
            if i != None:
                fp = self.mol2fp(i)
            else:
                fp = [0 for i in range(2048)]
            arr = np.vstack((arr, fp))
        df_ecfp6 = pd.DataFrame(np.asarray(arr).astype(int),columns=bit_headers)
        df_ecfp6.insert(loc=0, column= name, value=self.smiles)
        return df_ecfp6

def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']
    ecfp6 = ECFP6(exsample_smiles)
    res = ecfp6.compute_ECFP6('exsample')
    print(res)

if __name__ == '__main__':
    main()