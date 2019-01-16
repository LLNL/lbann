from proto import lbann_proto as lp
from proto import blocks as b

dl = 'data_parallel'
blocks = [3, 4, 6, 3]  # Blocks for ResNet-50.
bn_stats_aggregation = 'local'

# Create the layers.
input = lp.Input('data', dl, io_buffer='partitioned')
images = lp.Split('images', dl)
labels = lp.Split('labels', dl)
conv1 = b.ConvBNRelu2d('conv1', dl, 64, 7, stride=2, padding=3,
                       bn_stats_aggregation=bn_stats_aggregation)
pool1 = lp.Pooling('pool1', dl, num_dims=2, has_vectors=False, pool_dims_i=3,
                   pool_pads_i=1, pool_strides_i=2, pool_mode='max')
layer1 = b.ResBlock('block1', dl, blocks[0], 64, 256, stride=1,
                    bn_stats_aggregation=bn_stats_aggregation)
layer2 = b.ResBlock('block2', dl, blocks[1], 128, 512, stride=2,
                    bn_stats_aggregation=bn_stats_aggregation)
layer3 = b.ResBlock('block3', dl, blocks[2], 256, 1024, stride=2,
                    bn_stats_aggregation=bn_stats_aggregation)
layer4 = b.ResBlock('block4', dl, blocks[3], 512, 2048, stride=2,
                    bn_stats_aggregation=bn_stats_aggregation)
avgpool = lp.Pooling('avgpool', dl, num_dims=2, has_vectors=False, pool_dims_i=7,
                     pool_pads_i=0, pool_strides_i=1, pool_mode='average')
fc = lp.FullyConnected('fc1000', dl, num_neurons=1000, has_bias=False)
softmax = lp.Softmax('prob', dl)
ce = lp.CrossEntropy('cross_entropy', dl)
top1 = lp.CategoricalAccuracy('top1_accuracy', dl)
top5 = lp.TopKCategoricalAccuracy('top5_accuracy', dl, k=5)

# Set up the network.
images(input)
labels(input)
softmax(fc(avgpool(layer4(layer3(layer2(layer1(pool1(conv1(images)))))))))
ce(softmax)(labels)
top1(softmax)(labels)
top5(softmax)(labels)

# Explicitly set up weights for all layers.
weights = []  # For saving the non-batchnorm weights.
def setup_weights(l):
    if type(l) == lp.Convolution:
        w = l.add_weights(lp.HeNormalInitializer(), name_suffix='_kernel')
        weights.append(w)
    elif type(l) == lp.BatchNormalization:
        # Set the initial scale of the last BN of each residual block to be 0.
        # A bit hackish, this assumes the particular naming scheme.
        if l.name.endswith('_conv3_bn'):
            l.add_weights(lp.ConstantInitializer(value=0.0), name_suffix='_scale')
        else:
            l.add_weights(lp.ConstantInitializer(value=1.0), name_suffix='_scale')
        l.add_weights(lp.ConstantInitializer(value=0.0), name_suffix='_bias')
    elif type(l) == lp.FullyConnected:
        w = l.add_weights(lp.NormalInitializer(mean=0.0, standard_deviation=0.01))
        weights.append(w)
lp.traverse_model(input, setup_weights)

# Objective function/metrics.
obj = lp.ObjectiveFunction(ce, [lp.L2WeightRegularization(
    scale_factor=1e-4, weights=weights)])
top1_metric = lp.Metric('categorical accuracy', top1, '%')
top5_metric = lp.Metric('top-5 categorical accuracy', top5, '%')

lp.save_model('resnet50.prototext', input, dl, 256, 90, obj,
              metrics=[top1_metric, top5_metric],
              callbacks=[lp.CallbackPrint(), lp.CallbackTimer(),
                         lp.CallbackDropFixedLearningRate(
                             drop_epoch=[30, 60, 80], amt=0.1)])
