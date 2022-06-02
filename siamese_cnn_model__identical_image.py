from gc import callbacks
import os
from unittest.mock import call
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from models.siamese import SiameseCNN
from tools.data import Dataset
from tools.fscore import FScore

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="siamese_config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Init all classes
    dataset = Dataset(cfg.db)
    siameseCNN = SiameseCNN(cfg.model, dataset.shape)

    # Process dataset
    ds_train, ds_validation, ds_test = dataset.splitData()

    # Fit
    callbacks = siameseCNN.getCallBack(cfg.model)
    siameseCNN.model.fit(ds_train, epochs=cfg.model.hyperparameters.epochs, validation_data=ds_validation, callbacks=callbacks)

    # Test
    # Load best weights if callback checkpoint
    if(cfg.model.save.callback): siameseCNN.loadWeights(fr"{cfg.model.save.path}/weights")

    loss, accuracy, recall, precision = siameseCNN.model.evaluate(ds_test)
    f = FScore(precision, recall, cfg.model.metrics.fbeta)
    log.info(f"""
    loss: {loss}
    accuracy: {accuracy}
    recall: {recall}
    precision: {precision}
    f{cfg.model.metrics.fbeta} score: {f}
    """)

    # Save
    if(cfg.model.save.final):
        siameseCNN.save(cfg.model)


if __name__ == "__main__":
    my_app()