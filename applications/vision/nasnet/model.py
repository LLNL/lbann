from operations import *
from genotypes import *

import lbann.models
import lbann.models.resnet

class Cell(lbann.modules.Module):

   def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
       super().__init__()
       #print(C_prev_prev, C_prev, C)

       if reduction_prev:
           self.preprocess0 = FactorizedReduce(C_prev_prev, C)
       else:
           self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
       self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

       if reduction:
           op_names, indices = zip(*genotype.reduce)
           concat = genotype.reduce_concat
       else:
           op_names, indices = zip(*genotype.normal)
           concat = genotype.normal_concat
       self._compile(C, op_names, indices, concat, reduction)

   def _compile(self, C, op_names, indices, concat, reduction):
       assert len(op_names) == len(indices)
       self._steps = len(op_names) // 2
       self._concat = concat
       self.multiplier = len(concat)

       self._ops = []
       for name, index in zip(op_names, indices):
           stride = 2 if reduction and index < 2 else 1
           op = OPS[name](C, stride, True)
           self._ops += [op]
       self._indices = indices

   def forward(self, s0, s1): # add drop_prob later
       s0 = self.preprocess0(s0)
       s1 = self.preprocess1(s1)

       states = [s0, s1]
       for i in range(self._steps):
           h1 = states[self._indices[2 * i]]
           h2 = states[self._indices[2 * i + 1]]
           op1 = self._ops[2 * i]
           op2 = self._ops[2 * i + 1]
           h1 = op1(h1)
           h2 = op2(h2)
           s = h1 + h2
           states += [s]

       return lbann.Concatenation([states[i] for i in self._concat], dim=0)

class NetworkCIFAR(lbann.modules.Module):

   def __init__(self, C, num_classes, layers, auxiliary, genotype):
       super().__init__()
       self._layers = layers
       self._auxiliary = auxiliary

       stem_multiplier = 3
       C_curr = stem_multiplier * C
       self.stem1 = lbann.Convolution(num_dims = 2,
                                      num_output_channels = C_curr,
                                      conv_dims_i = 3,
                                      conv_pads_i = 1,
                                      has_bias = False)
       self.stem2 = lbann.BatchNormalization

       C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
       self.cells = []
       reduction_prev = False
       for i in range(layers):
           if i in [layers // 3, 2 * layers // 3]:
               C_curr *= 2
               reduction = True
           else:
               reduction = False

           cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

           reduction_prev = reduction
           self.cells += [cell]
           C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
           if i == 2 * layers // 3:
               C_to_auxiliary = C_prev

       self.global_pooling = lbann.Pooling(num_dims = 2,
                                           #adaptive avg - o/p 1
                                           pool_mode = "max")
       self.classifier = lbann.FullyConnected(num_neurons = num_classes)

   def forward(self, input):
       logits_aux = None
       s0 = s1 = self.stem1(input)
       #input2 = self.stem1(input)
       #s0 = s1 = self.stem2(input2)
       for i, cell in enumerate(self.cells):
           s0, s1 = s1, cell(s0, s1)
           if i == 2 * self._layers // 3:
               if self._auxiliary and self.training:
                   logits_aux = self.auxiliary_head(s1)
       out = self.global_pooling(s1)
         
       # will this work?
       size0 = out.size(0) 
       size1 = out.size(1)
       out = lbann.Reshape(out, dims='size0 size1')       

       logits = self.classifier(out)
       return logits, logits_aux

if __name__ == '__main__':
    
    genome = NASNet
    mymodel = NetworkCIFAR(32, 10, 20, False, genome) # nsga uses 34 instead of 32

    #myresnet = lbann.models.ResNet18

    input_ = lbann.Input(data_field='samples')
    x = lbann.Gaussian(neuron_dims='3 32 32')
    y = mymodel(x)
    #layers = list(lbann.traverse_layer_graph([x, input_]))
    
    print(y)
