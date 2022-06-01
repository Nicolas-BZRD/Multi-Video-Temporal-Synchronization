import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import hydra
from omegaconf import DictConfig, OmegaConf

from models.Siamese import SiameseCNN
from tools.data import Dataset

@hydra.main(version_base=None, config_path="configs", config_name="siamese_config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Init all classes
    dataset = Dataset(f"{os.getcwd()}/{cfg.db.path}")
    siameseCNN = SiameseCNN()

    # Process dataset
    ds_train, ds_validation, ds_test = dataset.splitData()

    # Fit
    siameseCNN.model.fit(ds_train, epochs=10, validation_data=ds_validation)


if __name__ == "__main__":
    my_app()