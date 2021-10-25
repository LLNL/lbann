from operations import *
from genotypes import *

class Cell(lbann.modules.Module):

   def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
       super().__init__()
       print(C_prev_prev, C_prev, C)

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

   def forward(self, s0, s1, drop_prob):
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
           if self.training and drop_prob > 0.:
               if not isinstance(op1, Identity):
                   h1 = drop_path(h1, drop_prob)
               if not isinstance(op2, Identity):
                   h2 = drop_path(h2, drop_prob)
           s = h1 + h2
           states += [s]

       return lbann.Concatenation([states[i] for i in self._concat], dim=0)
       #torch.cat([states[i] for i in self._concat], dim=1)

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
       input2 = self.stem1(input)
       s0 = s1 = self.stem(input2)
       for i, cell in enumerate(self.cells):
           s0, s1 = s1, cell(s0, s1, self.droprate)
           if i == 2 * self._layers // 3:
               if self._auxiliary and self.training:
                   logits_aux = self.auxiliary_head(s1)
       out = self.global_pooling(s1)
       logits = self.classifier(out.view(out.size(0), -1))
       return logits, logits_aux

if __name__ == '__main__':
    #import nasnet.genotypes as genotypes
    
    genome = NASNet
    model = NetworkCIFAR(34, 10, 20, False, genome)
    model.droprate = 0.0
