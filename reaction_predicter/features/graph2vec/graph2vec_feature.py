from rdkit import Chem
from rdkit import RDLogger
import networkx as nx
# from karateclub import Graph2Vec
import joblib
import pandas as pd

import pickle
import redis

import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.warning')

class Graph2VecPredicter:
    def __init__(self, molel_name, redis_client):

        def load_graph2vec_model():
            # # Попытка загрузить модель из кэша
            # cached_model = redis_client.get('graph2vec_model')
            # if cached_model:
            #     # Если модель найдена в кэше, десериализуем ее
            #     return pickle.loads(cached_model)
            # else:
            #     # Если модель не найдена, загрузите ее из файла или другого источника данных
            #     # и сохраните в кэш Redis
                model = joblib.load(molel_name)  # Пример загрузки модели
                # redis_client.set('graph2vec_model', pickle.dumps(model))
                return model

# Получение модели graph2vec
        self.graph2vec_model = load_graph2vec_model()
        
    @staticmethod    
    def mol_to_nx(mol):
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                    atomic_num=atom.GetAtomicNum(),
                    is_aromatic=atom.GetIsAromatic(),
                    atom_symbol=atom.GetSymbol())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=bond.GetBondType())

        return G
    
    def smiles_to_vec(self, smiles):

        smiles = list(smiles)
        mol = [Chem.MolFromSmiles(x) for x in smiles]
        graph = [self.mol_to_nx(x) for x in mol]
        
        hiv_graph2vec = self.graph2vec_model.infer(graph)
        
        return pd.DataFrame(hiv_graph2vec)
    
def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']

    graph2vec_obj = Graph2VecPredicter(
        './reaction_predicter/features/graph2vec/graph2vec_model.pkl'
        )
    print(graph2vec_obj.smiles_to_vec(exsample_smiles))

if __name__ == '__main__':
    main() 