
import os
import pandas as pd
import numpy as np
import yaml
import pickle
import logging
import click

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from utils.transformers import Selector, FillStringMissing, NormalizeLowerString, BoolHandler, FillNull, ConverteFloat
from utils.training_utils import find_specific_variables, get_features_attribute

import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--configfile', default='feature_config.yaml', help='Arquivo descritivo das features', type=str)
@click.option('--dataset_name', default='train.csv', help='Nome do dataset de treino', type=str)
def main(configfile, dataset_name):
    """Cria objetos de encoding para o modelo."""

    logger = logging.getLogger(__name__)

    logger.info('Iniciando a criação dos encoders')

    df = pd.read_csv(os.path.join('data', 'train_test', dataset_name))
    logger.info(f'Tabela de treino lida. Shape: {df.shape}')

    features = yaml.safe_load(open(os.path.join('src', 'config', configfile), 'r'))
    feature_target = find_specific_variables(features, 'target', specific_value=True)

    target_series = df[feature_target]

    try:
        feature_fs = yaml.safe_load(open(os.path.join('src', 'features', 'selected', 'features_selected.yaml')))
    except:
        logger.error('Não foi possível ler o arquivo de saída da etapa Feature Selection.')
        raise

    logger.info('Processando o objeto Selector 1')
    features_hard_remove = find_specific_variables(features, 'hard_remove', specific_value=True)
    features_default = find_specific_variables(features, 'descritiva', specific_value=True) + find_specific_variables(features, 'auxiliar', specific_value=True)
    features_engineering = find_specific_variables(features, 'cria_features', specific_value=True)
    features_selected = list((set(feature_fs["support_boruta"]) - set(features_hard_remove)) & set(features_default))
    features_selected_and_engineering = sorted(list(set(features_selected + features_engineering)))

    logger.info(f'Features do primeiro seletor: {features_selected_and_engineering}')
    seletor_1 = Selector(features=sorted(features_selected_and_engineering), target=feature_target)

    logger.info('Processando o objeto Selector 2')
    features_selected = sorted(list(set(feature_fs["support_boruta"]) - set(features_hard_remove)))
    logger.info(f'Features do segundo seletor: {features_selected}')
    seletor_2 = Selector(features=sorted(list(features_selected)), target=feature_target)
    df = seletor_2.transform(df)

    logger.info('Processando o objeto FillNulls')
    features_fill_null = find_specific_variables(features, 'fill_null', specific_value=True)
    features_fill_null = sorted(list(set(features_selected) & set(features_fill_null)))
    logger.info(f'Colunas a serem preenchidas com -999: {features_fill_null}')
    fill_null = FillNull(cols_to_adjust=features_fill_null)
    df = fill_null.transform(df)

    logger.info('Processando o objeto BoolHandler')
    colunas_bool = find_specific_variables(features, 'bool', specific_value=True)
    colunas_bool = sorted(list(set(features_selected) & set(colunas_bool)))
    logger.info(f'Colunas booleanas tratadas: {colunas_bool}')
    bool_handler = BoolHandler(cols_to_adjust=colunas_bool)
    df = bool_handler.transform(df)

    logger.info('Processando o objeto FillStringMissing')
    colunas_string = sorted(df.select_dtypes(include=['string', 'object']).columns.tolist())
    logger.info(f'Colunas strings preenchidas com valor `<vazio>` : {colunas_string}')
    fill_string_missing = FillStringMissing(cols_to_adjust=colunas_string)
    df = fill_string_missing.transform(df)

    logger.info('Processando o objeto NormalizeLowerString')
    cols_string_to_normalize = find_specific_variables(features, 'feature_to_normalize', specific_value=True)
    cols_string_to_normalize = sorted(list(set(features_selected) & set(cols_string_to_normalize)))
    logger.info(f'Colunas a serem normalizadas: {cols_string_to_normalize}')
    normalize_lower_string = NormalizeLowerString(cols_to_adjust=cols_string_to_normalize)
    df = normalize_lower_string.transform(df)

    logger.info(f'Colunas strings identificadas: {colunas_string}')
    logger.info('Processando o objeto de codificação')
    encoder = ColumnTransformer(
        [('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), colunas_string)],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    encoder.set_output(transform='pandas')
    df = encoder.fit_transform(df)

    logger.info('Processando o objeto ConverteFloat')
    conversor_float = ConverteFloat()
    df = conversor_float.transform(df)

    logger.info('Sucesso! Encoders criados')
    logger.info('Salvando os binários dos encoders')

    pickle.dump(seletor_1, open('models/encoders/seletor_1.pkl', 'wb'))
    pickle.dump(seletor_2, open('models/encoders/seletor_2.pkl', 'wb'))
    pickle.dump(fill_null, open('models/encoders/features_fill_null.pkl', 'wb'))
    pickle.dump(bool_handler, open('models/encoders/bool_handler.pkl', 'wb'))
    pickle.dump(fill_string_missing, open('models/encoders/fill_string_missing.pkl', 'wb'))
    pickle.dump(normalize_lower_string, open('models/encoders/normalize_lower_string.pkl', 'wb'))
    pickle.dump(encoder, open('models/encoders/encoder.pkl', 'wb'))
    pickle.dump(conversor_float, open('models/encoders/conversor_float.pkl', 'wb'))

    logger.info('Sucesso! Binários salvos')
    logger.info('Salvando a base encodada')

    df['y'] = target_series
    df.to_csv(os.path.join('data', 'train_test', 'train_encoded.csv'), index=False)

    logger.info('Sucesso! Base salva!')

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()