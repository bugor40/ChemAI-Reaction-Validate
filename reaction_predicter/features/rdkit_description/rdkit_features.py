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
    
class MACCS:
    def __init__(self, smiles):
        '''
        Класс расчитывает бинарный вектор длинной 167, который описывает молекулу

        smiles - список веществ 
        '''
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

class ECFP6:
    def __init__(self, smiles: list):
        '''
        Класс расчитывает бинарный вектор длинной 2048, который описывает молекулу

        smiles - список веществ 
        '''
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

    rdkit_2d = RDKit_2D(exsample_smiles)
    res = rdkit_2d.compute_2Drdkit('exsample')
    print(res)

    maccs = MACCS(exsample_smiles)
    res = maccs.compute_MACCS('exsample')
    print(res)
    
    
if __name__ == '__main__':
    main()
