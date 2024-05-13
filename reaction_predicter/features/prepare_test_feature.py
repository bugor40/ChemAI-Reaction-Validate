# подготовка данных 
import pandas as pd
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
    maccs_data =  maccs_builder.compute_MACCS('reactive')
    ecfp6_data =  ecfp6_builder.compute_ECFP6('reactive')
    graph2vec_data = graph2vec_builder.smiles_to_vec(smiles)
    mol2vec_data = mol2vec_builder.smiles_to_vec(smiles)
    roberta_data = roberta_builder.get_embeding(smiles)

    data = pd.concat([rdkit_data, maccs_data, ecfp6_data, graph2vec_data, mol2vec_data, roberta_data], axis=1)

    return data

def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']

    res = build_dataset(exsample_smiles)
    print(res)
    
    
if __name__ == '__main__':
    main()
