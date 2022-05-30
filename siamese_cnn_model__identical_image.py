import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from metaflow import FlowSpec, step
from model.Siamese import SiameseCNN

class FitFlow(FlowSpec):
    @step
    def start(self):
        self.siamese_cnn = SiameseCNN()
        self.next(self.data_process)

    @step
    def data_process(self):
        self.next(self.end)

    @step
    def end(self):
        print("this is the end !")


if __name__ == '__main__':
    FitFlow()