import lbann
from lbann.util import str_list

class GraphVertexData:
    def __init__(self, layers, num_features):
        """Object to hold list of layers, where each layer represents a vertex
           in a graph.

           Args:
               layers (iterator of layers): One dimensional iterator of node 
                                            features with N number of ndoes
               num_features (int) : the number of features per vertex
           
        """
        self.shape = (len(layers), num_features)
        self.layers = layers
        self.num_nodes = len(layers)
        self.num_features = num_features

    def __getitem__(self, node):
        """Get the feature vector of the None node represented as an LBANN layer
        
            args: node (int): The node to retrieve the features for. 

            returns: (Layer) : returns the features of the  Vertex <node> of  the graph.
                    
        """
        return self.layers[node]
    def __setitem__(self, node, feature):
        """Set the value of the row-th layer in 
           args: row (int):
                 layer (Layer): 
        """
        self.layers[node] = feature

    def size(index = None):
        """Get the size (shape) of the GraphVertexObject, where the size is represented
           as a tuple (n,m), where n is the number of nodes and m is the number of 
           features per node. 

           args: index (int): 0 to return the number of nodes and 1 to return the number of
                               features. 
           returns: (int) or (int,int): Either returns the tuple (n,m) or n or m. 

        """
        if (index):
            return self.shape[index]
        else:
            return self.shape

    def get_mat(self, cols = None):
        """Generates a matrix representation of the graph data.

           args: cols (int) 
        """

        mat = lbann.Identity(self.layers[0])
        
        for i in range(1,self.shape[0]):
            mat = lbann.Concatenation(mat, self.layers[i])

        if (cols):
            mat = lbann.Reshape(mat, dims=str_list([self.shape[0], cols]))    
        else:
            mat = lbann.Reshape(mat, dims=str_list([self.shape[0], self.shape[1]]))

        return mat
   
    def clone(self):
        """Generates a clone of the GraphVertexData object. Results in a 
           splitting in the DAG.
        """
        cloned_layers = [] 

        for i,node in enumerate(self.layers):
            temp = lbann.Split(node)
            layers[i] = lbann.Identity(temp)
            cloned_layers.append(lbann.Identity(temp))


        return GraphVertexData(cloned_layers, self.num_features)


    @classmethod
    def matrix_to_graph(cls, mat_layer, num_vertices, num_features):
        """Given a 2D matrix of shape (num_vertices, num_features), returns a 
           GraphVertexData object with num_vertices number of nodes with num_features. 
           
        """

        slice_points = str_list([i for i in range(0,num_vertices * num_features + 1, num_features)])
        flattened_layer = lbann.Reshape(mat_layer, dims = str(num_vertices * num_features))
        sliced_mat_layer = lbann.Slice(flattened_layer, axis = 0, slice_points = slice_points)

        list_of_layers = []
        for node in range(num_vertices):
            temp = lbann.Identity(sliced_mat_layer)
            list_of_layers.append(lbann.Reshape(temp, dims=str_list([1, num_features])))
        return cls(list_of_layers, num_features)
