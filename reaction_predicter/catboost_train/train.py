# обучение модели
from reaction_predicter.catboost_train.prepare_train_dataset import DataSet
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

def fit_model(redis_client, product):
    dataset_obj = DataSet(redis_client)
    X = dataset_obj.feature_prepare()
    y = dataset_obj.target_prepare(product)

    print(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

    clf = CatBoostClassifier(
        loss_function='Logloss',
        iterations=1000,
        depth= 5,  
        random_seed=42, 
        learning_rate=0.5, 
        custom_loss='AUC',
        eval_metric='AUC',
    )

    clf.fit(
        X = X_train,
        y = y_train,
        eval_set=(X_val, y_val),
        verbose = 1,
        early_stopping_rounds=300,
    )
    
    return clf