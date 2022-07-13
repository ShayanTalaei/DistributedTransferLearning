from Model import *
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

class Centralized_trainer():

    def __init__(self, client_count, dataset_factory, average_step_size = 1):
        initial_model = MNISTModel()
        train_dataset_factory = dataset_factory(True)
        test_dataset_factory  = dataset_factory(False)

        self.average_step_size = average_step_size
        self.server = Server(initial_model)    
    
        self.clients = []
        for id in range(client_count):
            train_set = train_dataset_factory(id)
            test_set  = test_dataset_factory(id)

            train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
            test_loader  = DataLoader(train_set, batch_size=1000, shuffle=False)
            
            client = Client(id, train_loader, test_loader, name = f"({2*id}, {2*id+1})", model = initial_model)
            self.clients.append(client)

    
    def train(lr_schedule):
        not_averaged_steps = 0
        
        for step, lr in lr_schedule:
            while not_averaged_steps < self.average_step_count:
                