import torch
from torch.fx.graph_module import GraphModule
import torch.nn as nn
import torch.nn.functional as F
import torch.fx

from module import *

class myCNN_24(nn.Module):

    def __init__(self, num_channels=1):
        super(myCNN_24, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.fc = nn.Linear(8 * 6 * 8, 27, bias = False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 8 * 6 * 8)
        x = self.fc(x)
        return x
    

class myCNN(nn.Module):

    def __init__(self, num_channels=1):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.fc = nn.Linear(8 * 8 * 8, 29, bias = False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 8 * 8 * 8)
        x = self.fc(x)
        return x
    

    def quantize(self, num_bits=4):
        graph_model = torch.fx.symbolic_trace(self)
        print(graph_model)
        # module quant
        graph_model = self._module_quant(graph_model, num_bits)
        print(graph_model)
        # function quant
        self._function_quant(graph_model, num_bits)
        print(graph_model)

    def _module_quant(self, graph_model: GraphModule, num_bits=8):
        device = list(graph_model.parameters())[0].device
        reassign = {}
        for i, (name, mod) in enumerate(graph_model.named_children()):
            qi = False
            qo = True
            if i == 0:
                qi = True
            if isinstance(mod, nn.Conv2d):
                reassign[name] = QConv2d(mod, qi, qo, num_bits).to(device)
            elif isinstance(mod, nn.Linear):
                reassign[name] = QLinear(mod, qi, qo, num_bits).to(device)
        
        for key, value in reassign.items():
            graph_model._modules[key] = value
        
        return graph_model
                
    def _function_quant(self, graph_model: GraphModule, num_bits=8):
        device = list(graph_model.parameters())[0].device
        nodes = list(graph_model.graph.nodes)
        for i, node in enumerate(nodes):
            if node.op == "call_function":
                if node.target == F.relu:
                    setattr(graph_model, "qrelu_%d" % i, QReLU().to(device))
                    with graph_model.graph.inserting_after(node):
                        new_node = graph_model.graph.call_module("qrelu_%d" % i, node.args, node.kwargs)
                        node.replace_all_uses_with(new_node)
                elif node.target == F.max_pool2d:
                    setattr(graph_model, "qmaxpool2d_%d" % i, QMaxPooling2d().to(device))
                    with graph_model.graph.inserting_after(node):
                        new_node = graph_model.graph.call_module("qmaxpool2d_%d" % i, node.args, node.kwargs)
                        node.replace_all_uses_with(new_node)
                graph_model.graph.erase_node(node)
        
        graph_model.recompile()
        return graph_model


if __name__ == "__main__":
    net = myCNN()
    net.quantize()