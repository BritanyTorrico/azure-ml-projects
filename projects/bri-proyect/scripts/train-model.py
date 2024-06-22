# importar librerías

import mlflow
import mlflow.sklearn
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import glob
import matplotlib.pyplot as plt

def main(args):
    # enable autologging
    mlflow.autolog()

    # leer datos
    df = get_data(args.training_data)

    # dividir datos
    X_train, X_test, y_train, y_test = split_data(df)
 #entrenar modelo
    model = train_model(X_train, y_train)

    # evaluar modelo
    eval_model(model, X_test, y_test)


# función que lee los datos
def get_data(data_path):
    all_files = glob.glob(data_path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
    return df

# función que divide los datos
def split_data(df):
    print("Dividiendo datos...")
    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]
    
    # codificar variables categóricas
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

# función que entrena el modelo
def train_model(X_train, y_train, ):
    print("Entrenando modelo...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    
    # Guardar el modelo en el path especificado
    mlflow.sklearn.save_model(model, args.model_output)
  

    return model

# función que evalúa el modelo
def eval_model(model, X_test, y_test):
    print("Evaluando modelo...")
    # calcular precisión
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Precisión:', acc)
   
    
    # calcular AUC-ROC
    y_probs = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_probs, multi_class='ovr')  # Para multiclase
    print('AUC-ROC:', auc)
    

    # Calcular F1-score
    f1 = f1_score(y_test, y_pred, average='macro')  # O 'micro' si prefieres
    print('F1-score:', f1)
    

    # Calcular precisión y recall
    precision = precision_score(y_test, y_pred, average='macro')  # O 'micro' si prefieres
    recall = recall_score(y_test, y_pred, average='macro')  # O 'micro' si prefieres
    print('Precisión:', precision)
    print('Recall:', recall)
  

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--model_output", dest='model_output',
                        type=str)
    # parse args
    args = parser.parse_args()

    # return args
    return args

# ejecutar el script
if __name__ == "__main__":
    # añadir espacio en los registros
    print("\n\n")
    print("*" * 60)

    # analizar argumentos
    args = parse_args()

    # ejecutar función principal
    main(args)

    # añadir espacio en los registros
    print("*" * 60)
    print("\n\n")
