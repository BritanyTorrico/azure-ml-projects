# importar librerías
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def main(args):
    # leer datos
    df = get_data(args.input_data)

    cleaned_data = clean_data(df)

    prepared_data = preprocess_data(cleaned_data)
    
    output_df = prepared_data.to_csv((Path(args.output_data) / "obesity.csv"), index=False)

# función que lee los datos
def get_data(path):
    df = pd.read_csv(path)

    # Contar las filas e imprimir el resultado
    row_count = len(df)
    print('Preparando {} filas de datos'.format(row_count))
    
    return df

# función que elimina valores nulos
def clean_data(df):
    df = df.dropna()
    return df

# función que preprocesa los datos
def preprocess_data(df):
    # Definir las columnas numéricas y categóricas
    numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

    # Definir los pasos del pipeline para las características numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())])

    # Definir los pasos del pipeline para las características categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combinar los pasos del pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Aplicar el preprocesamiento al DataFrame
    X_prepared = preprocessor.fit_transform(df)
    
    # Obtener los nombres de las características transformadas para las columnas categóricas
    cat_encoder = preprocessor.named_transformers_['cat']['onehot']
    
    # Construir los nombres de las características categóricas transformadas manualmente
    cat_feature_names = []
    for cat_feature in cat_encoder.categories_:
        cat_feature_names.extend([f"{cat_feature}_{value}" for value in cat_feature])
    
    # Combinar los nombres de las características numéricas y categóricas
    all_feature_names = numeric_features + cat_feature_names

    # Convertir los datos transformados a un DataFrame con los nombres de las columnas adecuadas
    X_prepared_df = pd.DataFrame(X_prepared, columns=all_feature_names)
    
    # Añadir la columna de destino 'NObeyesdad' al DataFrame preprocesado
    X_prepared_df['NObeyesdad'] = df['NObeyesdad'].values
    
    return X_prepared_df


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_data", dest='input_data', type=str)
    parser.add_argument("--output_data", dest='output_data', type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# ejecutar script
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
