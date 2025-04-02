import pandas as pd
import numpy as np

import os

import logging

import warnings
warnings.filterwarnings('ignore')


def main():
    logger = logging.getLogger(__name__)

    logger.info('Iniciando a criação da base interim.')

    logger.info('Leitura da base raw.')
    df = pd.read_csv(os.path.join('data', 'raw', 'MeliDataset.csv'), sep=";")
    logger.info(f'Sucesso! Base lida. Shape da base: {df.shape}')

    logger.info('Tratando dados duplicado')
    df = df.drop_duplicates()
    logger.info(f'Linhas duplicadas removidas. Shape: {df.shape}.')

    logger.info("Ajustando target para booleano.")
    df["y"] = (df["y"]=="yes").astype(int).fillna(0)

    logger.info("Ajustando valores da coluna 'default'.")
    if 'default' in df.columns:
        df['default'] = df['default'].replace({'yes': 'unknown'})
        logger.info("Coluna 'default' ajustada com sucesso: 'yes' -> 'unknown'.")
    else:
        logger.warning("Coluna 'default' não encontrada na base.")

    logger.info("Salvando a base interim.")
    path_output = os.path.join('data', 'interim', f'meli_interim.csv')
    df.to_csv(path_output, index=False)
    logger.info("Sucesso! Base interim salva!")


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()