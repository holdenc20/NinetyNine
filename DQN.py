import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, *, num_layers=3, hidden_dim=256, use_pooling=False, pooling_kernel=2):
        """Deep Q-Network PyTorch model with optional pooling and layer normalization."""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_pooling = use_pooling
        self.pooling_kernel = pooling_kernel

        layers = []
        input_dim = state_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Layer normalization for stabilization
            layers.append(nn.ReLU())
            if use_pooling:
                layers.append(nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel))
                input_dim = hidden_dim // pooling_kernel  # Update input size after pooling
            else:
                input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Identity())

        self.model = nn.Sequential(*layers)

    def forward(self, states) -> torch.Tensor:
        """Q function mapping from states to action-values."""
        if self.use_pooling:
            states = states.unsqueeze(-1)  # Add a channel dimension for pooling
        x = self.model(states)
        
        # Flatten the output if pooling is used
        if self.use_pooling:
            x = x.view(x.size(0), -1)  # Flatten the tensor to shape (batch_size, -1)

        return x

    @classmethod
    def custom_load(cls, data):
        model = cls(*data['args'], **data['kwargs'])
        model.load_state_dict(data['state_dict'])
        return model

    def custom_dump(self):
        return {
            'args': (self.state_dim, self.action_dim),
            'kwargs': {
                'num_layers': self.num_layers,
                'hidden_dim': self.hidden_dim,
                'use_pooling': self.use_pooling,
                'pooling_kernel': self.pooling_kernel,
            },
            'state_dict': self.state_dict(),
        }
