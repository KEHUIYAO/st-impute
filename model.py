import torch
import pytorch_lightning as pl

class MLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        
        # Initialize the model parameters using He initialization
        for layer in self.layers:
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    