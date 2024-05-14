# подготовка данных 
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from reaction_predicter.features.rdkit_description.rdkit_features import RDKit_2D, MACCS, ECFP6
from reaction_predicter.features.graph2vec.graph2vec_feature import Graph2VecPredicter
from reaction_predicter.features.mol2vec.mol2vec_feature import Mol2VecCals
from reaction_predicter.features.roBERTa.roberta_feature import roBERTaCals

def build_dataset(
        smiles: list,
        redis_client,
):

    rdkit_builder = RDKit_2D(smiles)
    maccs_builder = MACCS(smiles)
    ecfp6_builder = ECFP6(smiles)
    graph2vec_builder = Graph2VecPredicter('./reaction_predicter/features/graph2vec/graph2vec_model.pkl', redis_client)
    mol2vec_builder = Mol2VecCals('./reaction_predicter/features/mol2vec/model_300dim.pkl', redis_client)
    roberta_builder = roBERTaCals()

    rdkit_data = rdkit_builder.compute_2Drdkit('reactive')
    rdkit_data.columns = ['rdkit_2d_'+i for i in rdkit_data.columns]

    maccs_data =  maccs_builder.compute_MACCS('reactive')
    maccs_data.columns = ['maccs_'+i for i in maccs_data.columns]

    ecfp6_data =  ecfp6_builder.compute_ECFP6('reactive')
    ecfp6_data.columns = ['ecpf6_'+i for i in ecfp6_data.columns]

    graph2vec_data = graph2vec_builder.smiles_to_vec(smiles)
    graph2vec_data.columns = ['graph2vec_'+str(i) for i in graph2vec_data.columns]

    mol2vec_data = mol2vec_builder.smiles2vec(smiles)
    mol2vec_data.columns = ['mol2vec_'+str(i) for i in mol2vec_data.columns]

    roberta_data = roberta_builder.get_embeding(smiles)
    roberta_data.columns = ['roBERTa_'+str(i) for i in roberta_data.columns]

    data = pd.concat([rdkit_data, maccs_data, graph2vec_data, mol2vec_data, roberta_data, ecfp6_data], axis=1).astype('float32')

    with open('reaction_predicter/features/scaler.pkl','rb') as f:
        scaler = pickle.load(f)

    with open('reaction_predicter/features/pca.pkl','rb') as f:
        pca = pickle.load(f)

    data_scaler = scaler.transform(data)
    data_pca = pca.transform(data_scaler)

    data_pca = pd.DataFrame(data_pca)

    data_mean = pd.DataFrame(data_pca.mean(axis = 0)).T
    
    return data_mean

def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']

    res = build_dataset(exsample_smiles)
    print(res)
    
    
if __name__ == '__main__':
    main()
