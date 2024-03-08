"""Implements an Adapter"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


### HyperNet
class TaskEmbeddingHyperNet(nn.Module):
    """The hypernet is to generate task embedding I_t; h_I in the paper"""
    def __init__(self, config):
        super().__init__()
        self.activation = nn.ReLU()
        self.linear = nn.Linear(config.task_embedding_input_size, config.task_hidden_size)
        self.project = nn.Linear(config.task_hidden_size, config.task_embedding_size)

    def forward(self, task_embedding):
        x = self.linear(task_embedding)
        x = self.activation(x)
        x = self.project(x)
        return x

class TaskConditionalHyperNet(nn.Module):
    """Generates task conditional adapter weights/bias; W_U and W_D in the paper"""
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_generator = nn.Linear(config.task_embedding_size, \
                                          self.input_dim * self.output_dim)
        self.bias_generator = nn.Linear(config.task_embedding_size, self.output_dim)

    def forward(self, embeddings):
        weight = self.weight_generator(embeddings).view(self.output_dim, self.input_dim)
        bias = self.bias_generator(embeddings).view(-1)
        return weight, bias

class LayerNormHyperNet(nn.Module):
    """Generate layer normalization weights/bais; W_gama, W_beta in the paper"""
    def __init__(self, config, input_dim):
        super().__init__()
        self.task_embedding_size = config.task_embedding_size
        self.weight_generator = nn.Linear(self.task_embedding_size, input_dim)
        self.bias_generator = nn.Linear(self.task_embedding_size, input_dim)

    def forward(self, input):
        return self.weight_generator(input), self.bias_generator(input)

class AdapterHyperNet(nn.Module):
    """Generate three hypernet: task embedding hypernet, task conditional hypernet and layer norm hypernet"""
    def __init__(self, config, input_dim):
        super().__init__()
        self.device = config.device
        self.task_embedding_size = config.task_embedding_size
        self.adapter_hidden_size = config.adapter_hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.task_names = config.task_names
        self.layer_id_embeddings = nn.Embedding(self.num_hidden_layers,
                                                self.task_embedding_size)
        self.position_id_embeddings = nn.Embedding(2, self.task_embedding_size)
        # Init task to embeddings dictionary
        self.task_to_embeddings = {}
        for task in self.task_names.split(','):
            self.task_to_embeddings[task] = nn.Parameter(torch.Tensor(torch.randn(self.task_embedding_size)))
        
        # Hypernet
        self.task_embedding_hypernet = TaskEmbeddingHyperNet(config)
        self.down_sampler_hypernet = TaskConditionalHyperNet(config, input_dim, self.adapter_hidden_size)
        self.up_sampler_hypernet = TaskConditionalHyperNet(config, self.adapter_hidden_size, input_dim)
        self.layer_norm_hypernet = LayerNormHyperNet(config, input_dim)
    
    def get_embedding(self, task_name, layer_id, position_id):
        task_embedding = self.task_to_embeddings[task_name]
        layer_id_tensor = torch.tensor([layer_id], dtype=torch.long)
        layer_embedding = self.layer_id_embeddings(layer_id_tensor).to(self.device)
        type_id_tensor = torch.tensor([position_id], dtype=torch.long)
        type_embedding = self.position_id_embeddings(type_id_tensor).to(self.device)
        embedding = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1), type_embedding.view(1, -1)],
                               axis=0)
        embedding = self.task_embedding_hypernet(embedding.view(-1))
        return embedding
    
    def forward(self, task_name, layer_id, position_id):
        ### position_id = 0, self attention adapter
        ### position_id = 1, feed forward adapter
        embeddings = self.get_embedding(task_name, layer_id, position_id)

        down_sampler_weight, down_sampler_bias = self.down_sampler_hypernet(embeddings)
        up_sampler_weight, up_sampler_bias = self.up_sampler_hypernet(embeddings)
        layer_norm_weight, layer_norm_bias = self.layer_norm_hypernet(embeddings)

        return AdapterHyperNetOuput(down_sampler_weight, down_sampler_bias,
                                    up_sampler_weight, up_sampler_bias,
                                    layer_norm_weight, layer_norm_bias)

## HyperNet output
@dataclass
class AdapterHyperNetOuput:
    down_sampler_weight: torch.FloatTensor = None
    down_sampler_bias: torch.FloatTensor = None
    up_sampler_weight: torch.FloatTensor = None
    up_sampler_bias: torch.FloatTensor = None
    layer_norm_weight: torch.FloatTensor = None
    layer_norm_bias: torch.FloatTensor = None

### Adapter class
class BasicAdapter(nn.Module):

    def __init__(self, config, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.adapter_hidden_size = config.adapter_hidden_size
        self.activation = nn.GELU()
        self.down_sampler = nn.Linear(self.input_dim, self.adapter_hidden_size)
        self.up_sampler = nn.Linear(self.adapter_hidden_size, self.input_dim)

    def forward(self, input):
        x = self.down_sampler(input)
        x = self.activation(x)
        x = self.up_sampler(x)
        return x + input


class HyperAdapter(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.activation = nn.GELU()
        self.input_dim = input_dim
        self.adapter_hypernet = AdapterHyperNet(config, input_dim)
        self.enable_adapter_layer_norm = config.enable_adapter_layer_norm

    def forward(self, input, task_name, layer_id, position_id):
        hypernet_output = self.adapter_hypernet.forward(task_name, layer_id, position_id)
        x = F.linear(input, weight=hypernet_output.down_sampler_weight,
                        bias=hypernet_output.down_sampler_bias)
        
        x = self.activation(x)
        x = F.linear(x, weight=hypernet_output.up_sampler_weight,
                          bias=hypernet_output.up_sampler_bias)
        
        if self.enable_adapter_layer_norm:
            x = F.layer_norm(x, (self.input_dim,),
                                weight=hypernet_output.layer_norm_weight,
                                bias=hypernet_output.layer_norm_bias)
        
        return x + input

        
