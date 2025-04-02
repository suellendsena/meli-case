
import os
import pandas as pd
import logging
import click 

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


@click.command()
@click.option('--dataset_name', default='meli_processed.csv', help='Nome do dataset processado', type=str)
def main(dataset_name):
    logger = logging.getLogger(__name__)

    logger.info('Iniciando o split da base - Classificação promoção')
    df = pd.read_csv(os.path.join('data', 'processed', dataset_name))
    logger.info(f'Shape da base: {df.shape}')

    df_train, df_test, y_train, y_test = train_test_split(
        df.drop(['y'], axis=1), 
        df.y, 
        test_size = 0.2, 
        random_state = 96
    )

    df_train = pd.concat([df_train, y_train.rename('y')], axis=1)
    df_test = pd.concat([df_test, y_test.rename('y')], axis=1)

    logger.info(f'Salvando a base de treino. Shape {df_train.shape}.')
    df_train.to_csv(os.path.join('data', 'train_test', 'train.csv'), index=False)

    logger.info(f'Salvando a base de teste. Shape {df_test.shape}.')
    df_test.to_csv(os.path.join('data', 'train_test', 'test.csv'), index=False)

    logger.info(f'Sucesso! Base para o treino do modelo salva!')


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()