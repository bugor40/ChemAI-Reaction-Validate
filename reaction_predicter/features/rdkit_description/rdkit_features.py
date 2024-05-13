import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class DescriptionCalc:
    def __init__(self, smiles):
        '''
        Класс расчитывает физико-хмические признаки молекулы, которые описывают молекулу

        smiles - список веществ 
        '''
        self.mols = []
        for i in tqdm(smiles):
            try:
                self.mols.append(Chem.MolFromSmiles(i))
            except:
                self.mols.append(None)
                
        self.smiles = smiles

class RDKit_2D(DescriptionCalc):
    def __init__(self, smiles):
        super().__init__(smiles)

    def compute_2Drdkit(self, name):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header = calc.GetDescriptorNames()
        
        def calculate_descriptors(mol):
            if mol is not None:
                return calc.CalcDescriptors(mol)
            else:
                return [None] * len(header)
            
        rdkit_2d_desc = [calculate_descriptors(mol) for mol in tqdm(self.mols)]
        
        df = pd.DataFrame(rdkit_2d_desc, columns=header)
        return df
    
class MACCS(DescriptionCalc):
    def __init__(self, smiles):
        super().__init__(smiles)

    def compute_MACCS(self, name):
        MACCS_list = []
        header = ['bit' + str(i) for i in range(167)]
        
        def calculate_descriptors(mol):
            if mol != None:
                return list(MACCSkeys.GenMACCSKeys(mol).ToBitString())
            else:
                return [0 for i in range(167)]

        MACCS_list = [calculate_descriptors(mol) for mol in tqdm(self.mols)]
        
        df = pd.DataFrame(MACCS_list,columns=header)
        return df

class ECFP6(DescriptionCalc):
    def __init__(self, smiles):
        super().__init__(smiles)

    def mol2fp(self, mol, nBits, radius = 3):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits=nBits)
        return fp

    def compute_ECFP6(self, name, nBits = 2048):
        bit_headers = ['bit' + str(i) for i in range(nBits)]
        arr = lil_matrix((len(self.mols), nBits), dtype=np.int8)
        
        for idx, mol in enumerate(tqdm(self.mols)):
            if mol != None:
                fp = self.mol2fp(mol, nBits)
                on_bits = fp.GetOnBits()

                for bit in on_bits:
                    arr[idx, bit] = 1
                    
        df_ecfp6 = pd.DataFrame.sparse.from_spmatrix(arr, columns=bit_headers)
        return df_ecfp6
    
def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']

    ecfp6 = ECFP6(exsample_smiles)
    res = ecfp6.compute_ECFP6('exsample', nBits = 2048)
    print(res)

    rdkit_2d = RDKit_2D(exsample_smiles)
    res = rdkit_2d.compute_2Drdkit('exsample')
    print(res)

    maccs = MACCS(exsample_smiles)
    res = maccs.compute_MACCS('exsample')
    print(res)
    
    
if __name__ == '__main__':
    main()
