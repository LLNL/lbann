import abc
from lbann import optimizers_pb2
import lbann.core.util

#============================================================================
class NoOptimizer :
  '''
  An Optimizer that does nothing. Used in testing.
  Possibly useful elsewhere, e.g, as a placehoder during model development
  '''
#  def __init__(self) :
#    pass

  def export_proto(self) :
    '''Construct and return a protobuf message.'''
    proto = metrics_pb2.Optimizer.NoOptimizer()
    return proto

#============================================================================
class AdaGrad :
  '''
  AdaGrad Optimizer

  Reference:

  John Duchi, Elad Hazan, and Yoram Singer. "Adaptive subgradient
  methods for online learning and stochastic optimization." Journal
  of Machine Learning Research 12, no. Jul (2011): 2121-2159.

  :param: learn_rate <double> required
  :param: eps <double> optional; default: 1e-8
  '''

  def __init__(self, learn_rate, eps=1e-8) :
    self._learn_rate = learn_rate
    self._eps = eps
    

  def export_proto(self) :
    '''Construct and return a protobuf message.'''
    proto = metrics_pb2.Optimizer.AdaGrad()
    proto.learn_rate = self._learn_rate
    proto.eps = self._eps
    return proto

#============================================================================
class Adam :
  '''
  Adam Optimizer

  Reference:

  Diederik P. Kingma and Jimmy Ba. "Adam: A method for stochastic
  optimization." arXiv preprint arXiv:1412.6980 (2014).

  :param: learn_rate <double> required
  :param: beta1 <double> optional; default: 0.9
  :param: beta2 <double> optional; default: 0.99
  :param: eps <double> optional; default: 1e-8
  '''

  def __init__(self, learn_rate, beta1=0.9, beta2=0.99, eps=1e-8) :
    self._learn_rate = learn_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._eps = eps

  def export_proto(self) :
    '''Construct and return a protobuf message.'''
    proto = metrics_pb2.Optimizer.Adam()
    proto.learn_rate = self._learn_rate
    proto.beta1 = self._beta1
    proto.beta2 = self._beta2
    proto.eps = self._eps
    return proto

#============================================================================
class HypergradientAdam :
  '''
  HypergradientAdam Optimizer

  Reference:

  Baydin et al. "Online Learning Rate Adaptation with Hypergradient
  Descent", 2017.

  :param: init_learning_rate <double> optional; default: 1e-3; initial Adam learning rate
  :param: hyper_learning_rate <double> optional; default: 1e-7; Hypergradient learning rate
  :param: beta1 <double> optional; default: 0.9; decay rate for the first moment moving average
  :param: beta2 <double> optional; default: 0.99; decay rate for the second moment moving average
  :param: eps <double> optional; default: 1e-8; small factor to avoid division by zero
  '''

  def __init__(self, init_learning_rate=1e-3, hyper_learning_rate=1e-7, beta1=0.9, beta2=0.99, eps=1e-8) :
    self._init_learning_rate = init_learning_rate
    self._hyper_learning_rate = hyper_learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._eps = eps

  def export_proto(self) :
    '''Construct and return a protobuf message.'''
    proto = metrics_pb2.Optimizer.HypergradientAdam()
    proto.init_learning_rate = self._init_learning_rate
    proto.hyper_learning_rate = self._hyper_learning_rate
    proto.beta1 = self._beta1
    proto.beta2 = self._beta2
    proto.eps = self._eps
    return proto

#============================================================================
class RMSprop :
  '''
  RMSprop Optimizer

  `See < https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_

  :param: learn_rate <double> required
  :param: decay_rate <double> required
  :param: eps <double> optional; default: 1e-8
  '''

  def __init__(self) :
    self._learn_rate = learn_rate
    self._decay_rate = decay_rate
    self._eps = eps

  def export_proto(self) :
    '''Construct and return a protobuf message.'''
    proto = metrics_pb2.Optimizer.RMSprop()
    proto.learn_rate = self._learn_rate
    proto.decay_rate = self._decay_rate
    proto.eps = self._eps
    return proto

#============================================================================
class SGD :
  '''
  Stochastic gradient descent Optimizer

  :param: learn_rate <double> optional; default: 0.0; decay rate for gradient accumulation; a momentum of zero corresponds to vanilla SGD
  :param: momentum <double> optional; default: Set to zero for vanilla SGD
  :param: nesterov <bool> optional; default: false; controls whether Nesterov acceleration is applied
  '''

  def __init__(self, learn_rate, momentum, nesterov) :
    self._learn_rate = learn_rate
    self._momentum = momentum
    self._nesterov = nesterov

  def export_proto(self) :
    '''Construct and return a protobuf message.'''
    proto = metrics_pb2.Optimizer.SGD()
    proto.learn_rate = self._learn_rate
    proto.momentum = self._momentum
    proto.nesterov = self._nesterov
    return proto

#============================================================================
