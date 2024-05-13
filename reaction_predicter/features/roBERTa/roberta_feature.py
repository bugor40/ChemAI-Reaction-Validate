from rdkit import Chem
from rdkit import RDLogger

import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.warning')

class roBERTaCals:
    def __init__(self):
        self.model = SentenceTransformer('all-roberta-large-v1')

    def get_embeding(self, smiles):
        
        smiles = list(smiles)
        
        def to_inchi(mol_smiles):
            mol = Chem.MolFromSmiles(mol_smiles)
            inchistr = Chem.MolToInchi(mol)
            
            return inchistr
        
        inchi_list = [to_inchi(mol_smiles) for mol_smiles in tqdm(smiles)]
        
        embed = self.model.encode(inchi_list, show_progress_bar = True)
        header = [i for i in range(embed.shape[1])]
        
        return pd.DataFrame(embed, columns = header)


def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']

    roberta_obj = roBERTaCals()
    print(roberta_obj.get_embeding(exsample_smiles))

if __name__ == '__main__':
    main()  