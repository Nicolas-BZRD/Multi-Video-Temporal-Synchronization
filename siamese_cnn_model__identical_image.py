import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

    # Init train and test dataset
    dataset = Dataset(cfg.db.path_train, (cfg.db.img_heigth, cfg.db.img_width), cfg.db.grayscale)
    dataset_test = Dataset(cfg.db.path_test, (cfg.db.img_heigth, cfg.db.img_width), cfg.db.grayscale)

    # Process dataset
    ds_train, ds_validation = dataset.splitData()

    # Init model
    dropout = [cfg.model.dropout.first, cfg.model.dropout.second, cfg.model.dropout.third, cfg.model.dropout.fourth]
    siameseCNN = SiameseCNN(dropout, dataset.shape)

    # Process model
    callbacks = []
    if cfg.model.callback.checkpoint: callbacks.append(siameseCNN.checkpointCallback('weights'))

    # Fit
    if(cfg.model.other.fit):
        siameseCNN.model.fit(ds_train, epochs=cfg.model.hyperparameters.epochs, validation_data=ds_validation, callbacks=callbacks)

    # Test
    if(cfg.model.other.load_weights):
        siameseCNN.loadWeights(cfg.model.other.path_weights)

    dataset_test.data = dataset_test.data.batch(1)
    loss, accuracy, recall, precision = siameseCNN.model.evaluate(dataset_test.data)
    f = FScore(precision, recall, cfg.model.metrics.fbeta)

    log.info(f"loss: {loss}\naccuracy: {accuracy}\nrecall: {recall}\nprecision: {precision}\nf{cfg.model.metrics.fbeta} score: {f}")

    # Save
    if(cfg.model.other.save):
        siameseCNN.save(cfg.model.other.path_save)


if __name__ == "__main__":
    my_app()