"""Neural network modules for transformer models."""
import math
import lbann
from lbann.modules.base import Module, FullyConnectedModule
from lbann.util import make_iterable


class MultiheadAttention(Module):
    """Parallel instances of scaled dot-product attention.

    See:

    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
    "Attention is all you need." In Advances in Neural Information
    Processing Systems, pp. 5998-6008. 2017.

    Args:
        embed_dim (int): Size of representation space.
        num_heads (int): Number of parallel attention instances. Must
            evenly divide `embed_dim`.
        name (str): Default name is in the form
            'multiheadattention<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, embed_dim, num_heads, branches, d_kv=None, name=None):
        super().__init__()
        MultiheadAttention.global_count += 1
        self.instance = 0
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if d_kv == None:
            self.inner_dim = embed_dim
            self.head_dim = embed_dim // num_heads
        else:
            self.inner_dim = d_kv * num_heads
            self.head_dim = d_kv

        if branches == 0:
            self.ENABLE_SUBGRAPH = False
            self.BRANCHES = 0
        else:
            self.ENABLE_SUBGRAPH = True
            self.BRANCHES = branches

        # Module name
        self.name = name
        if not self.name:
            self.name = f"multiheadattention{MultiheadAttention.global_count}"

        # Weights for fully-connected layers
        self.query_weights = [
            lbann.Weights(
                initializer=lbann.GlorotNormalInitializer(),
                name=f"{self.name}_query_matrix",
            ),
            lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0),
                name=f"{self.name}_query_bias",
            ),
        ]
        self.key_weights = [
            lbann.Weights(
                initializer=lbann.GlorotNormalInitializer(),
                name=f"{self.name}_key_matrix",
            ),
            lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0),
                name=f"{self.name}_key_bias",
            ),
        ]
        self.value_weights = [
            lbann.Weights(
                initializer=lbann.GlorotNormalInitializer(),
                name=f"{self.name}_value_matrix",
            ),
            lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0),
                name=f"{self.name}_value_bias",
            ),
        ]

        self.output_weights = [
            lbann.Weights(
                initializer=lbann.GlorotNormalInitializer(),
                name=f"{self.name}_output_matrix",
            ),
            lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0),
                name=f"{self.name}_output_bias",
            ),
        ]

    def forward(self, queries, keys, values, mask=None):
        """Apply multi-head attention.

        The input and output tensors are interpreted as sequences of
        vectors, where the first tensor dimension is the sequence
        dimension.

        Args:
            queries (lbann.Layer): Sequence of query vectors.
            keys (lbann.Layer): Sequence of key vectors.
            values (lbann.Layer): Sequence of value vectors.
            mask (lbann.Layer, optional): Additive attention mask. If
                the (i,j) entry is very negative (e.g. -1e9), then the
                ith query does not attend to the jth key/value pair.

        Returns:
            lbann.Layer: Sequence of output vectors. The sequence
                length is the same as `queries`.

        """
        ENABLE_SUBGRAPH = self.ENABLE_SUBGRAPH
        BRANCHES = self.BRANCHES
        if ENABLE_SUBGRAPH:
            if self.num_heads % BRANCHES != 0:
                raise ValueError("Num heads should be divisible by BRANCHES")
        self.instance += 1
        name = f"{self.name}_instance{self.instance}"

        # Apply fully-connected layers to input sequences
        queries_fc = lbann.ChannelwiseFullyConnected(
            queries,
            weights=self.query_weights,
            output_channel_dims=[self.inner_dim],
            name=f"{name}_queries_fc",
        )
        keys_fc = lbann.ChannelwiseFullyConnected(
            keys,
            weights=self.key_weights,
            output_channel_dims=[self.inner_dim],
            name=f"{name}_keys_fc",
        )
        values_fc = lbann.ChannelwiseFullyConnected(
            values,
            weights=self.value_weights,
            output_channel_dims=[self.inner_dim],
            name=f"{name}_values_fc",
        )

        # Slice embedding vectors for each head
        slice_points = [self.head_dim * i for i in range(self.num_heads + 1)]
        queries_slice = lbann.Slice(
            queries_fc,
            axis=1,
            slice_points=slice_points,
            name=f"{name}_queries_slice",
            parallel_strategy={"grid_tag": 0},
        )
        keys_slice = lbann.Slice(
            keys_fc,
            axis=1,
            slice_points=slice_points,
            name=f"{name}_keys_slice",
            parallel_strategy={"grid_tag": 0},
        )
        values_slice = lbann.Slice(
            values_fc,
            axis=1,
            slice_points=slice_points,
            name=f"{name}_values_slice",
            parallel_strategy={"grid_tag": 0},
        )

        # Compute scaled dot-product attention for each head
        attentions = []
        tag = 0
        for head in range(self.num_heads):
            head_name = f"{name}_myattention_head{head}"

            # Attention inputs

            if ENABLE_SUBGRAPH:
                if head % int(self.num_heads / BRANCHES) == 0:
                    tag += 1

                q = lbann.Identity(queries_slice, parallel_strategy={"grid_tag": tag})
                k = lbann.Identity(keys_slice, parallel_strategy={"grid_tag": tag})
                v = lbann.Identity(values_slice, parallel_strategy={"grid_tag": tag})
            else:
                q = lbann.Identity(queries_slice)
                k = lbann.Identity(keys_slice)
                v = lbann.Identity(values_slice)

            # Multiply queries and keys
            # Note: num_queries x num_keys
            y = lbann.MatMul(
                q,
                k,
                transpose_b=True,
                name=f"{head_name}_matmul",
            )
            y = lbann.WeightedSum(
                y,
                scaling_factors=1 / math.sqrt(self.head_dim),
                name=f"{head_name}_scale",
            )

            if ENABLE_SUBGRAPH:
                if mask != None:
                    y = lbann.Sum([y, mask[tag]], name=f"{head_name}_mask")
            else:
                if mask:
                    y = lbann.Sum([y, mask], name=f"{head_name}_mask")
            y = lbann.ChannelwiseSoftmax(y, name=f"{head_name}_softmax")

            # Attention output
            # Note: num_queries x head_dim

            attentions.append(lbann.MatMul(y, v, name=head_name))

            # Strong scaling

        # Concatenate heads and apply fully-connected layer
        if ENABLE_SUBGRAPH:
            attentions = lbann.Concatenation(
                attentions,
                axis=1,
                name=f"{name}_heads_concat",
                parallel_strategy={"grid_tag": 0},
            )
        else:
            attentions = lbann.Concatenation(
                attentions,
                axis=1,
                name=f"{name}_heads_concat",
            )

        outputs_fc = lbann.ChannelwiseFullyConnected(
            attentions,
            weights=self.output_weights,
            output_channel_dims=[self.embed_dim],
            name=f"{name}",
        )
        return outputs_fc


class MultiheadAttentionAllSubGraph(Module):
    """Parallel instances of scaled dot-product attention.

    See:

    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
    "Attention is all you need." In Advances in Neural Information
    Processing Systems, pp. 5998-6008. 2017.

    Args:
        embed_dim (int): Size of representation space.
        num_heads (int): Number of parallel attention instances. Must
            evenly divide `embed_dim`.
        name (str): Default name is in the form
            'multiheadattention<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, embed_dim, num_heads, branches, d_kv=None, name=None):
        super().__init__()
        MultiheadAttention.global_count += 1
        self.instance = 0
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if d_kv == None:
            self.inner_dim = embed_dim
            self.head_dim = embed_dim // num_heads
        else:
            self.inner_dim = d_kv * num_heads
            self.head_dim = d_kv

        if branches == 0:
            self.ENABLE_SUBGRAPH = False
            self.BRANCHES = 0
        else:
            self.ENABLE_SUBGRAPH = True
            self.BRANCHES = branches

        # Module name
        self.name = name
        if not self.name:
            self.name = f"multiheadattention{MultiheadAttention.global_count}"

        # Weights for fully-connected layers
        self.query_weights = [
            lbann.Weights(
                initializer=lbann.GlorotNormalInitializer(),
                name=f"{self.name}_query_matrix",
            ),
            lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0),
                name=f"{self.name}_query_bias",
            ),
        ]
        self.key_weights = [
            lbann.Weights(
                initializer=lbann.GlorotNormalInitializer(),
                name=f"{self.name}_key_matrix",
            ),
            lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0),
                name=f"{self.name}_key_bias",
            ),
        ]
        self.value_weights = [
            lbann.Weights(
                initializer=lbann.GlorotNormalInitializer(),
                name=f"{self.name}_value_matrix",
            ),
            lbann.Weights(
                initializer=lbann.ConstantInitializer(value=0),
                name=f"{self.name}_value_bias",
            ),
        ]

        # Channelwise FC in SubGraph
        self.output_weights = []

        for head in range(branches):
            self.output_weights.append(
                [
                    lbann.Weights(
                        initializer=lbann.GlorotNormalInitializer(),
                        name=f"{self.name}_head{head}_output_matrix",
                    ),
                    lbann.Weights(
                        initializer=lbann.ConstantInitializer(value=0),
                        name=f"{self.name}_head{head}_output_bias",
                    ),
                ]
            )

    def forward(self, queries, keys, values, mask=None):
        """Apply multi-head attention.

        The input and output tensors are interpreted as sequences of
        vectors, where the first tensor dimension is the sequence
        dimension.

        Args:
            queries (lbann.Layer): Sequence of query vectors.
            keys (lbann.Layer): Sequence of key vectors.
            values (lbann.Layer): Sequence of value vectors.
            mask (lbann.Layer, optional): Additive attention mask. If
                the (i,j) entry is very negative (e.g. -1e9), then the
                ith query does not attend to the jth key/value pair.

        Returns:
            lbann.Layer: Sequence of output vectors. The sequence
                length is the same as `queries`.

        """
        ENABLE_SUBGRAPH = self.ENABLE_SUBGRAPH
        BRANCHES = self.BRANCHES
        if ENABLE_SUBGRAPH:
            if self.num_heads % BRANCHES != 0:
                raise ValueError("Num heads should be divisible by BRANCHES")
        self.instance += 1
        name = f"{self.name}_instance{self.instance}"

        # Apply fully-connected layers to input sequences
        queries_fc = lbann.ChannelwiseFullyConnected(
            queries,
            weights=self.query_weights,
            output_channel_dims=[self.inner_dim],
            name=f"{name}_queries_fc",
        )
        keys_fc = lbann.ChannelwiseFullyConnected(
            keys,
            weights=self.key_weights,
            output_channel_dims=[self.inner_dim],
            name=f"{name}_keys_fc",
        )
        values_fc = lbann.ChannelwiseFullyConnected(
            values,
            weights=self.value_weights,
            output_channel_dims=[self.inner_dim],
            name=f"{name}_values_fc",
        )

        # Slice embedding vectors for each head
        slice_points = [self.head_dim * i for i in range(self.num_heads + 1)]
        queries_slice = lbann.Slice(
            queries_fc,
            axis=1,
            slice_points=slice_points,
            name=f"{name}_queries_slice",
            parallel_strategy={"grid_tag": 0},
        )
        keys_slice = lbann.Slice(
            keys_fc,
            axis=1,
            slice_points=slice_points,
            name=f"{name}_keys_slice",
            parallel_strategy={"grid_tag": 0},
        )
        values_slice = lbann.Slice(
            values_fc,
            axis=1,
            slice_points=slice_points,
            name=f"{name}_values_slice",
            parallel_strategy={"grid_tag": 0},
        )

        # Compute scaled dot-product attention for each head
        attentions = []

        # variable to combine heads locally in sub-grids
        temp_attentions = []
        tag = 0
        for head in range(self.num_heads):
            head_name = f"{name}_myattention_head{head}"

            # Attention inputs

            if ENABLE_SUBGRAPH:
                if head % int(self.num_heads / BRANCHES) == 0:
                    temp_attentions.append([])
                    tag += 1

                q = lbann.Identity(queries_slice, parallel_strategy={"grid_tag": tag})
                k = lbann.Identity(keys_slice, parallel_strategy={"grid_tag": tag})
                v = lbann.Identity(values_slice, parallel_strategy={"grid_tag": tag})
            else:
                q = lbann.Identity(queries_slice)
                k = lbann.Identity(keys_slice)
                v = lbann.Identity(values_slice)

            # Multiply queries and keys
            # Note: num_queries x num_keys
            y = lbann.MatMul(
                q,
                k,
                transpose_b=True,
                name=f"{head_name}_matmul",
            )
            y = lbann.WeightedSum(
                y,
                scaling_factors=1 / math.sqrt(self.head_dim),
                name=f"{head_name}_scale",
            )

            if ENABLE_SUBGRAPH:
                if mask != None:
                    y = lbann.Sum([y, mask[tag]], name=f"{head_name}_mask")
            else:
                if mask:
                    y = lbann.Sum([y, mask], name=f"{head_name}_mask")
            y = lbann.ChannelwiseSoftmax(y, name=f"{head_name}_softmax")

            # Attention output
            # Note: num_queries x head_dim
            y = lbann.MatMul(y, v, name=head_name)
            # attentions.append(lbann.MatMul(y, v, name=head_name))

            temp_attentions[-1].append(y)

        for count, temp_attention in enumerate(temp_attentions):
            if self.BRANCHES == self.num_heads:
                # No need to concat the heads at subgrid level
                # if number of subgrids is equal to number of heads
                attention_single_subgrid = temp_attentions[count][0]
            else:
                attention_single_subgrid = lbann.Concatenation(
                    temp_attention, axis=1, name=f"{name}_subgrid_heads_concat{count}"
                )

            attention_single_subgrid = lbann.ChannelwiseFullyConnected(
                attention_single_subgrid,
                weights=self.output_weights[count],
                output_channel_dims=[self.embed_dim],
                name=f"{name}_cfc_{count}",
            )

            attentions.append(attention_single_subgrid)

        # Strong scaling

        grid_sum_slice = lbann.Cross_Grid_Sum_Slice(attentions)

        attentions = []

        for head in range(self.BRANCHES):
            attentions.append(
                lbann.Identity(grid_sum_slice, parallel_strategy={"grid_tag": head + 1})
            )

        return attentions


class MultiheadAttentionAllSubGraphInputSubGrids(Module):
    """Parallel instances of scaled dot-product attention.

    See:

    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
    Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
    "Attention is all you need." In Advances in Neural Information
    Processing Systems, pp. 5998-6008. 2017.


    This module expects inputs in subgrids format
    if number of heads is 16 and subgrids is 4
    then input should be a list of legnth 4

    Args:
        embed_dim (int): Size of representation space.
        num_heads (int): Number of parallel attention instances. Must
            evenly divide `embed_dim`.
        name (str): Default name is in the form
            'multiheadattention<index>'.

    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, embed_dim, num_heads, branches, d_kv=None, name=None):
        super().__init__()
        MultiheadAttention.global_count += 1
        self.instance = 0
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if d_kv == None:
            self.inner_dim = embed_dim
            self.head_dim = embed_dim // num_heads
        else:
            self.inner_dim = d_kv * num_heads
            self.head_dim = d_kv

        if branches == 0:
            self.ENABLE_SUBGRAPH = False
            self.BRANCHES = 0
        else:
            self.ENABLE_SUBGRAPH = True
            self.BRANCHES = branches

        # Module name
        self.name = name
        if not self.name:
            self.name = f"multiheadattention{MultiheadAttention.global_count}"

        # Weights for fully-connected layers
        self.query_weights = []
        self.key_weights = []
        self.value_weights = []

        for head in range(branches):
            self.query_weights.append(
                [
                    lbann.Weights(
                        initializer=lbann.GlorotNormalInitializer(),
                        name=f"{self.name}_head{head}_query_matrix",
                    ),
                    lbann.Weights(
                        initializer=lbann.ConstantInitializer(value=0),
                        name=f"{self.name}_head{head}_query_bias",
                    ),
                ]
            )
            self.key_weights.append(
                [
                    lbann.Weights(
                        initializer=lbann.GlorotNormalInitializer(),
                        name=f"{self.name}_head{head}_key_matrix",
                    ),
                    lbann.Weights(
                        initializer=lbann.ConstantInitializer(value=0),
                        name=f"{self.name}_head{head}_key_bias",
                    ),
                ]
            )
            self.value_weights.append(
                [
                    lbann.Weights(
                        initializer=lbann.GlorotNormalInitializer(),
                        name=f"{self.name}_head{head}_value_matrix",
                    ),
                    lbann.Weights(
                        initializer=lbann.ConstantInitializer(value=0),
                        name=f"{self.name}_head{head}_value_bias",
                    ),
                ]
            )

        # Channelwise FC in SubGraph
        self.output_weights = []

        for head in range(branches):
            self.output_weights.append(
                [
                    lbann.Weights(
                        initializer=lbann.GlorotNormalInitializer(),
                        name=f"{self.name}_head{head}_output_matrix",
                    ),
                    lbann.Weights(
                        initializer=lbann.ConstantInitializer(value=0),
                        name=f"{self.name}_head{head}_output_bias",
                    ),
                ]
            )

    def forward(self, queries, keys, values, mask=None):
        """Apply multi-head attention.

        The input and output tensors are interpreted as sequences of
        vectors, where the first tensor dimension is the sequence
        dimension.

        Args:
            queries (lbann.Layer): Sequence of query vectors.
            keys (lbann.Layer): Sequence of key vectors.
            values (lbann.Layer): Sequence of value vectors.
            mask (lbann.Layer, optional): Additive attention mask. If
                the (i,j) entry is very negative (e.g. -1e9), then the
                ith query does not attend to the jth key/value pair.

        Returns:
            lbann.Layer: Sequence of output vectors. The sequence
                length is the same as `queries`.

        """
        ENABLE_SUBGRAPH = self.ENABLE_SUBGRAPH
        BRANCHES = self.BRANCHES
        if ENABLE_SUBGRAPH:
            if self.num_heads % BRANCHES != 0:
                raise ValueError("Num heads should be divisible by BRANCHES")
        self.instance += 1
        name = f"{self.name}_instance{self.instance}"

        # Apply fully-connected layers to input sequences
        queries_fc = []
        keys_fc = []
        values_fc = []

        # Slice embedding vectors for each head
        slice_points = [
            self.head_dim * i for i in range(int(self.num_heads / self.BRANCHES) + 1)
        ]

        # Queries strong scaling in CFC
        attentions = []
        for count, query in enumerate(queries):
            temp = lbann.ChannelwiseFullyConnected(
                query,
                weights=self.query_weights[count],
                output_channel_dims=[self.inner_dim],
                name=f"{name}_subgrid{count}_queries_fc",
            )
            attentions.append(temp)

        grid_sum_slice = lbann.Cross_Grid_Sum_Slice(attentions)

        attentions = []

        for head in range(self.BRANCHES):
            attentions.append(
                lbann.Identity(grid_sum_slice, parallel_strategy={"grid_tag": head + 1})
            )

        for head in range(self.BRANCHES):
            temp = lbann.Slice(
                attentions[head],
                axis=1,
                slice_points=slice_points,
                name=f"{name}_subgrid{head}_queries_slice",
            )

            queries_fc.append(temp)

        # keys strong scaling in CFC

        attentions = []
        for count, key in enumerate(keys):
            temp = lbann.ChannelwiseFullyConnected(
                key,
                weights=self.key_weights[count],
                output_channel_dims=[self.inner_dim],
                name=f"{name}_subgrid{count}_keys_fc",
            )

            attentions.append(temp)

        grid_sum_slice = lbann.Cross_Grid_Sum_Slice(attentions)

        attentions = []

        for head in range(self.BRANCHES):
            attentions.append(
                lbann.Identity(grid_sum_slice, parallel_strategy={"grid_tag": head + 1})
            )

        for head in range(self.BRANCHES):
            temp = lbann.Slice(
                attentions[head],
                axis=1,
                slice_points=slice_points,
                name=f"{name}_subgrid{head}_keys_slice",
            )

            keys_fc.append(temp)

        # Values strong scaling in CFC
        attentions = []

        for count, value in enumerate(values):
            temp = lbann.ChannelwiseFullyConnected(
                value,
                weights=self.value_weights[count],
                output_channel_dims=[self.inner_dim],
                name=f"{name}_subgrid{count}_values_fc",
            )
            attentions.append(temp)

        grid_sum_slice = lbann.Cross_Grid_Sum_Slice(attentions)

        attentions = []

        for head in range(self.BRANCHES):
            attentions.append(
                lbann.Identity(grid_sum_slice, parallel_strategy={"grid_tag": head + 1})
            )

        for head in range(self.BRANCHES):
            temp = lbann.Slice(
                attentions[head],
                axis=1,
                slice_points=slice_points,
                name=f"{name}_subgrid{head}_values_slice",
            )
            values_fc.append(temp)

        queries_slice = []
        keys_slice = []
        values_slice = []

        for branch in range(self.BRANCHES):
            querie_slice = queries_fc[branch]
            key_slice = keys_fc[branch]
            value_slice = values_fc[branch]

            for head in range(int(self.num_heads / self.BRANCHES)):
                queries_slice.append(lbann.Identity(querie_slice))
                keys_slice.append(lbann.Identity(key_slice))
                values_slice.append(lbann.Identity(value_slice))

        # Compute scaled dot-product attention for each head
        attentions = []

        # variable to combine heads locally in sub-grids
        temp_attentions = []
        tag = 0
        for head in range(self.num_heads):
            head_name = f"{name}_myattention_head{head}"

            # Attention inputs
            if head % int(self.num_heads / BRANCHES) == 0:
                temp_attentions.append([])
                tag += 1

            q = lbann.Identity(queries_slice[head])
            k = lbann.Identity(keys_slice[head])
            v = lbann.Identity(values_slice[head])

            # Multiply queries and keys
            # Note: num_queries x num_keys
            y = lbann.MatMul(
                q,
                k,
                transpose_b=True,
                name=f"{head_name}_matmul",
            )
            y = lbann.WeightedSum(
                y,
                scaling_factors=1 / math.sqrt(self.head_dim),
                name=f"{head_name}_scale",
            )

            if ENABLE_SUBGRAPH:
                if mask != None:
                    y = lbann.Sum([y, mask[tag]], name=f"{head_name}_mask")
            else:
                if mask:
                    y = lbann.Sum([y, mask], name=f"{head_name}_mask")
            y = lbann.ChannelwiseSoftmax(y, name=f"{head_name}_softmax")

            # Attention output
            # Note: num_queries x head_dim
            y = lbann.MatMul(y, v, name=head_name)
            # attentions.append(lbann.MatMul(y, v, name=head_name))

            temp_attentions[-1].append(y)

        for count, temp_attention in enumerate(temp_attentions):
            if self.BRANCHES == self.num_heads:
                # No need to concat the heads at subgrid level
                # if number of subgrids is equal to number of heads
                attention_single_subgrid = temp_attentions[count][0]
            else:
                attention_single_subgrid = lbann.Concatenation(
                    temp_attention, axis=1, name=f"{name}_subgrid_heads_concat{count}"
                )

            attention_single_subgrid = lbann.ChannelwiseFullyConnected(
                attention_single_subgrid,
                weights=self.output_weights[count],
                output_channel_dims=[self.embed_dim],
                name=f"{name}_cfc_{count}",
            )

            attentions.append(attention_single_subgrid)

        # Strong scaling

        grid_sum_slice = lbann.Cross_Grid_Sum_Slice(attentions)

        attentions = []

        for head in range(self.BRANCHES):
            attentions.append(
                lbann.Identity(grid_sum_slice, parallel_strategy={"grid_tag": head + 1})
            )

        return attentions
