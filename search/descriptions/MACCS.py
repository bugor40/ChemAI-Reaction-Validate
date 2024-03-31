import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


class MACCS:
    def __init__(self, smiles):

        self.mols = []
        for i in tqdm(smiles):
            try:
                self.mols.append(Chem.MolFromSmiles(i))
            except:
                self.mols.append(None)
        self.smiles = smiles

    def compute_MACCS(self, name):
        MACCS_list = []
        header = ['bit' + str(i) for i in range(167)]
        for i in tqdm(range(len(self.mols))):
            if self.mols[i] != None:
                ds = list(MACCSkeys.GenMACCSKeys(self.mols[i]).ToBitString())
            else:
                ds = [0 for i in range(167)]
                
            MACCS_list.append(ds)
        df = pd.DataFrame(MACCS_list,columns=header)
        df.insert(loc=0, column= name, value=self.smiles)
        return df

def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']
    maccs = MACCS(exsample_smiles)
    res = maccs.compute_MACCS('exsample')
    print(res)

if __name__ == '__main__':
    main()