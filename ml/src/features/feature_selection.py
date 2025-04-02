import os
import pandas as pd
import yaml
import logging
import click 

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

from utils.transformers import BoolHandler
from utils.training_utils import find_specific_variables, get_features_attribute

import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--configfile', default='feature_config.yaml', help='Arquivo descritivo das features', type=str)
@click.option('--dataset_name', default='train.csv', help='Nome do dataset de treino', type=str)
def main(configfile, dataset_name):

    logger = logging.getLogger(__name__)

    logger.info('Iniciando o Feature Selection')
    
    features = yaml.safe_load(open(os.path.join('src', 'config', configfile), 'r'))

    logger.info(f'Lendo a base de treino')
    df = pd.read_csv(os.path.join('data', 'train_test', dataset_name))
    logger.info(f'Sucessos. Shape: {df.shape}')

    logger.info('Removendo features auxiliares e hard-remove')
    features_auxiliares = find_specific_variables(features, 'auxiliar', specific_value=True)
    features_hard_remove = find_specific_variables(features, 'hard_remove', specific_value=True)
    features_auxiliares_remaining = list((set(features_auxiliares + features_hard_remove)) & set(df.columns))
    logger.info(f'Features a serem removidas: {features_auxiliares_remaining}')
    df = df.drop(columns=features_auxiliares_remaining, axis=1)
    logger.info(f'Sucesso! Shape: {df.shape}')


    logger.info('Tratando colunas booleanas')
    colunas_bool = find_specific_variables(features, 'bool', specific_value=True)
    colunas_bool = sorted(list(set(colunas_bool) & set(df.columns)))
    bool_handler = BoolHandler(cols_to_adjust=colunas_bool)
    df = bool_handler.transform(df)


    logger.info("Encoding das features string para mapeamento numérico")
    colunas_string = df.columns[
        (df.dtypes == 'object') | (df.dtypes == 'string')
    ].to_list()
    df[colunas_string] = df[colunas_string].apply(lambda x: pd.factorize(x)[0])
    

    logger.info("Iniciando o Boruta")
    df = df[sorted(df.columns)]
    logger.info(df.columns)

    model = RandomForestClassifier(max_depth=6, min_samples_leaf=100, class_weight= 'balanced', n_jobs=-1, random_state=96)
    feat_selector = BorutaPy(
        model,
        n_estimators=60,
        verbose=2,
        random_state=98,
        max_iter=40,
        alpha=0.01
    )

    feature_target = find_specific_variables(features, 'target', specific_value=True)
    feat_selector.fit(
        df.drop(feature_target, axis=1).values,
        df[feature_target].values.ravel()
    )

    features_selecionadas = df.drop(feature_target, axis=1).columns[feat_selector.support_].to_list()
    features_rejeitadas = df.drop(feature_target, axis=1).columns[~feat_selector.support_].to_list()

    fs_dict = {
        'support_boruta': features_selecionadas,
        'rejected_boruta': features_rejeitadas
    }

    logger.info(f'Sucesso! Boruta executado. {len(features_selecionadas)} variáveis selecionadas.')
    with open(os.path.join('src', 'features', 'selected', 'features_selected.yaml'), 'w') as output:
        yaml.dump(fs_dict, output)


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
