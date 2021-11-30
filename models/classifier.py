from OpenAttack.classifier import Classifier
from typing import List, Union
from .model import BERT, LSTM
import numpy as np

class MyClassifier(Classifier):
    def __init__(self, model: Union[BERT, LSTM]):
       self.model = model 

    def get_prob(self, input_):
        rt = []
        for sent in input_:
            prob = self.model.predict(sent)
            rt.append(np.array(prob.squeeze().tolist()))

        return np.array(rt)

