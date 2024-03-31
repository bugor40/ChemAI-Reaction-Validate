import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class RDKit_2D:
    def __init__(self, smiles):
        self.mols = []
        for i in tqdm(smiles):
            try:
                self.mols.append(Chem.MolFromSmiles(i))
            except:
                self.mols.append(None)
                
        self.smiles = smiles

    def compute_2Drdkit(self, name):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header = calc.GetDescriptorNames()
        for i in tqdm(range(len(self.mols))):
            if self.mols[i] != None:
                ds = calc.CalcDescriptors(self.mols[i])
            else:
                ds = [None for i in range(len(Descriptors._descList))]
            rdkit_2d_desc.append(ds)
        df = pd.DataFrame(rdkit_2d_desc,columns=header)
        df.insert(loc=0, column= name, value=self.smiles)
        return df

def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']
    rdkit_2d = RDKit_2D(exsample_smiles)
    res = rdkit_2d.compute_2Drdkit('exsample')
    print(res)

if __name__ == '__main__':
    main()