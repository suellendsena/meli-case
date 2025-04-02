import os
import pickle
import logging
import click
from sklearn.pipeline import Pipeline
from utils.transformers import BuildFeatures, Json2DF

import warnings
warnings.filterwarnings('ignore')


@click.command()
def main():
    """Cria o binário final da pipeline para deploy."""
    logger = logging.getLogger(__name__)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger.info('Iniciando criação da pipeline final do modelo.')

    encoders_path_folder = 'models/encoders'

    # Transformadores fixos
    json_to_df = Json2DF()
    cria_features = BuildFeatures()

    logger.info('Lendo binários dos transformadores.')
    seletor_1 = pickle.load(open(os.path.join(encoders_path_folder, 'seletor_1.pkl'), 'rb'))
    seletor_2 = pickle.load(open(os.path.join(encoders_path_folder, 'seletor_2.pkl'), 'rb'))
    fill_null = pickle.load(open(os.path.join(encoders_path_folder, 'features_fill_null.pkl'), 'rb'))
    bool_handler = pickle.load(open(os.path.join(encoders_path_folder, 'bool_handler.pkl'), 'rb'))
    fill_string_missing = pickle.load(open(os.path.join(encoders_path_folder, 'fill_string_missing.pkl'), 'rb'))
    normalize_lower_string = pickle.load(open(os.path.join(encoders_path_folder, 'normalize_lower_string.pkl'), 'rb'))
    encoder = pickle.load(open(os.path.join(encoders_path_folder, 'encoder.pkl'), 'rb'))
    conversor_float = pickle.load(open(os.path.join(encoders_path_folder, 'conversor_float.pkl'), 'rb'))

    logger.info('Lendo modelo preditivo.')
    model_path = os.path.join('models', 'predictors', 'model.pkl')
    modelo = pickle.load(open(model_path, 'rb'))

    logger.info('Montando pipeline final.')

    pipeline_list = [
        ('json_to_df', json_to_df),
        ('seletor_1', seletor_1),
        ('cria_features', cria_features),
        ('seletor_2', seletor_2),
        ('fill_null', fill_null),
        ('bool_encoder', bool_handler),
        ('fill_string_missing', fill_string_missing),
        ('normalize_lower_string', normalize_lower_string),
        ('encoder', encoder),
        ('conversor_float', conversor_float),
        ('seletor_3', seletor_2),
        ('modelo', modelo)
    ]

    pipeline_prod = Pipeline(steps=pipeline_list)

    logger.info('Exportando pipeline final.')
    os.makedirs('models/wrapped', exist_ok=True)
    pickle.dump(
        pipeline_prod,
        open(os.path.join('models', 'wrapped', 'model_pipeline_prod.pkl'), 'wb')
    )

    logger.info('Sucesso! Artefato final exportado para deploy.')


if __name__ == '__main__':
    main()
