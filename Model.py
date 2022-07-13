from distutils import core
import torch.nn
import torch


class MNISTModel(nn.Module):
    def __init__(self):
        self.super() 
        self.l1=torch.nn.Linear(28*28, 20)
        self.l2=torch.nn.Linear(20, 2)
        self.loss=torch.nn.CrossEntropyLoss

    def forward(self, x):
        x=torch.nn.Flatten(x)
        x=self.l1(x)
        x=self.l2(x)
        return x

    def compute_loss_batch(self, X_b, y_b):
        y_hat=self.forward(X_b)
        return self.loss(y_hat, y_b)

    
    def evaluate(self, loader):
        with torch.no_grad():
            self.eval()
            loss_sum=0
            correct=0
            for x_b,y_b in loader:
                # x, y=x.cuda(), y.cuda()
                y_hat=self.forward(x_b).detach()
                loss_sum+=self.loss(y_hat, y_b)
                y_pred=torch.argmax(y_hat, dim=1) # need double check 
                correct+=torch.sum(y_pred == y_b)
            self.train()

            return loss_sum/len(loader), correct/len(loader)
