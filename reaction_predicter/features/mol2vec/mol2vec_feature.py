from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence
from tqdm import tqdm

import pickle
import redis

# import numpy/pandas ---------------------------------------------------------
import numpy as np
import pandas as pd

class Mol2VecCals:
    def __init__(self, molel_name, redis_client):

        def load_mol2vec_model():
            # Попытка загрузить модель из кэша
            cached_model = redis_client.get('mol2vec_model')
            if cached_model:
                # Если модель найдена в кэше, десериализуем ее
                return pickle.loads(cached_model)
            else:
                # Если модель не найдена, загрузите ее из файла или другого источника данных
                # и сохраните в кэш Redis
                model = word2vec.Word2Vec.load(molel_name)
                # Пример загрузки модели
                redis_client.set('mol2vec_model', pickle.dumps(model))
                return model

        self.w2v_model = load_mol2vec_model()

    def smiles2vec(self, smiles):
        
        smiles = list(smiles)
        mol = [Chem.MolFromSmiles(x) for x in tqdm(smiles)]
        sentence = [MolSentence(mol2alt_sentence(x, radius=1)) for x in tqdm(mol)]
        
        def sentences2vec(sentences, model, unseen=None):
            """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
            sum of vectors for individual words.

            Parameters
            ----------
            sentences : list, array
                List with sentences
            model : word2vec.Word2Vec
                Gensim word2vec model
            unseen : None, str
                Keyword for unseen words. If None, those words are skipped.
                https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

            Returns
            -------
            np.array
            """
            keys = set(model.wv.key_to_index.keys())
            vec = []
            if unseen:
                unseen_vec = model.wv.word_vec(unseen)

            for sentence in tqdm(sentences):
                if unseen:
                    vec.append(sum([model.wv.word_vec(y) if y in set(sentence) & keys
                            else unseen_vec for y in sentence]))
                else:
                    vec.append(sum([model.wv.word_vec(y) for y in sentence
                                    if y in set(sentence) & keys]))
            return np.array(vec)
        
        
        embeddings = np.array([vec for vec in sentences2vec(sentence, self.w2v_model, unseen='UNK')])
        
        return pd.DataFrame(embeddings)


def main():
    exsample_smiles = ['C1CCCCC1', 'c1ccccc1']

    mol2vec_obj = Mol2VecCals('./reaction_predicter/features/mol2vec/model_300dim.pkl')
    print(mol2vec_obj.smiles2vec(exsample_smiles))


if __name__ == '__main__':
    main()  
    