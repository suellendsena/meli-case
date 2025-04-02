import pandas as pd
import os

import logging
from utils.transformers import BuildFeatures

import warnings
warnings.filterwarnings('ignore')

def main():
    logger = logging.getLogger(__name__)

    logger.info('Iniciando a criação das features.')
    df = pd.read_csv(os.path.join('data', 'interim', 'meli_interim.csv'))
    logger.info(f'Shape da base: {df.shape}.')

    criador_features = BuildFeatures()

    logger.info('Iniciando processamento.')
    df = criador_features.transform(df)
    logger.info(f'Sucesso! Processamento finalizado. Shape da tabela processada: {df.shape}')


    logger.info(f"Salvando a base processed.")
    path_output = os.path.join('data', 'processed', f'meli_processed.csv')
    df.to_csv(path_output, index=False)
    logger.info(f"Sucesso! Base processed salva!")


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()