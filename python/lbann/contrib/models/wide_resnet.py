import lbann.models.resnet

class WideResNet50_2(lbann.models.resnet.ResNet):
    """Wide ResNet-50-2 neural network.

    See:
        Sergey Zagoruyko and Nikos Komodakis. "Wide Residual Networks."
        In Proceedings of the British Machine Vision Conference. 2016.

    """

    global_count = 0  # Static counter, used for default names.

    def __init__(self, output_size,
                 zero_init_residual=True,
                 bn_stats_aggregation='local',
                 name=None):
        """Initialize WRN-50-2.

        Args:
            output_size (int): Size of output tensor.
            zero_init_residual (bool, optional): Whether to initialize
                the final batch normalization in residual branches
                with zeros.
            bn_stats_aggregation (str, optional): Aggregation mode for
                batch normalization statistics.
            name (str, optional): Module name.
                (default: 'wide_resnet50_module<index>')

        """
        WideResNet50_2.global_count += 1
        name = name or 'wide_resnet50_module{}'.format(
            WideResNet50_2.global_count)
        super().__init__(lbann.models.resnet.BottleneckBlock,
                         output_size, (3,4,6,3), (64,128,256,512),
                         zero_init_residual, bn_stats_aggregation, name,
                         width=2)
