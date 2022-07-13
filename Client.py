import torch
from copy import deepcopy

class Client():


    def __init__(self, id, train_loader, test_loader, name = None, model = None):
        self.id = id
        self.name = name

        self.model = deepcopy(model)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def take_step(self, steps):
        taken_steps = 0
        for X_b, y_b in self.train_loader:
            pass