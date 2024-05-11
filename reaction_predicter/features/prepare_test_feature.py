# подготовка данных 
import pandas as pd
from reaction_predicter.features.rdkit_description.rdkit_features import RDKit_2D, MACCS, ECFP6

def build_dataset(smiles: list):

    rdkit_builder = RDKit_2D(smiles)
    maccs_builder = MACCS(smiles)
    ecfp6_builder = ECFP6(smiles)

    rdkit_data = rdkit_builder.compute_2Drdkit('reactive')
    maccs_data =  maccs_builder.compute_MACCS('reactive')
    ecfp6_data =  ecfp6_builder.compute_ECFP6('reactive')

    data = pd.concat([rdkit_data, maccs_data, ecfp6_data], axis=1)

    return data

def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']

    res = build_dataset(exsample_smiles)
    print(res)
    
    
if __name__ == '__main__':
    main()
