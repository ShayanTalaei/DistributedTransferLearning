import torch

class Server():


    def __init__(self):
        self.model = None
        self.received_state_dicts = []

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def average_received_models(self):
        pass

    