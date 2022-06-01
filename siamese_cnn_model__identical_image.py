import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import hydra
from omegaconf import DictConfig, OmegaConf

from models.siamese import SiameseCNN
from tools.data import Dataset

@hydra.main(version_base=None, config_path="configs", config_name="siamese_config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Init all classes
    dataset = Dataset(cfg.db)
    siameseCNN = SiameseCNN(cfg.model, dataset.shape)
    # Process dataset
    ds_train, ds_validation, ds_test = dataset.splitData()

    # Fit
    siameseCNN.model.fit(ds_train, epochs=cfg.model.hyperparameters.epochs, validation_data=ds_validation)

    # Test
    result = siameseCNN.model.evaluate(ds_test)


if __name__ == "__main__":
    my_app()