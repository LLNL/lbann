import lbann 
from lbann.util import str_list

class lbann_Data_Mat:
    def __init__(self, list_of_layers, layer_size):
        self.shape = (len(list_of_layers), layer_size)
        self.layers = list_of_layers
    
    def __getitem__(self, row):
        # To Do: Add bounds checking 
        return self.layers[row]
    def __setitem__(self, row, layer):
        # To Do: Add bounds checking
        self.layers[row] = layer

    def get_mat(self, cols = None):
        

        mat = lbann.Identity(self.layers[0])
        
        for i in range(1,self.shape[0]):
            mat = lbann.Concatenation(mat, self.layers[i])

        if (cols):
            mat = lbann.Reshape(mat, dims=str_list([self.shape[0], cols]))    
        else:
            mat = lbann.Reshape(mat, dims=str_list([self.shape[0], self.shape[1]]))

        return mat
    @classmethod
    def mat_to_data(cls, mat_layer, num_vertices, out_channels):
        
        slice_points = str_list([i for i in range(0,num_vertices * out_channels + 1, out_channels)])
        flattened_layer = lbann.Reshape(mat_layer, dims = str(num_vertices * out_channels))
        sliced_mat_layer = lbann.Slice(flattened_layer, axis = 0, slice_points = slice_points)

        list_of_layers = []
        for node in range(num_vertices):
            temp = lbann.Identity(sliced_mat_layer)
            list_of_layers.append(lbann.Reshape(temp, dims=str_list([1, out_channels])))
        return cls(list_of_layers, out_channels)

class lbann_Graph_Data:
    def __init__(self, input_layer, num_vertices, num_features, num_classes):
        self.num_vertices = num_vertices 
        self.num_features = num_features 
        self.num_classes = num_classes 

        self.x, self.adj, self.y = self.gen_data(input_layer)

    def generate_slice_points (self):
        slice_points = [i for i in range(0,self.num_vertices * self.num_features + 1, self.num_features)]
        adj_mat = slice_points[-1] + self.num_vertices * self.num_vertices 
        slice_points.append(adj_mat)
        targets = slice_points[-1] + self.num_classes 
        slice_points.append(targets)
        return str_list(slice_points)
    
    def gen_data(self, input_layer):
        slice_points = self.generate_slice_points()
        sliced_graph  = lbann.Slice(input_layer, axis = 0, slice_points = slice_points, name="Sliced_Input")

        node_features = [] 

        for i in range(self.num_vertices):
            temp = lbann.Identity(sliced_graph)
            node_features.append(lbann.Reshape(temp, dims=str_list([1,self.num_features])))

        
        adj_mat_in = lbann.Identity(sliced_graph) 
        adj_mat = lbann.Reshape(adj_mat_in, dims = str_list([self.num_vertices, self.num_vertices])) 

        y = lbann.Identity(sliced_graph)
        y = lbann.Reshape(y, dims=str(self.num_classes))
        
        x = lbann_Data_Mat(node_features, self.num_features)
        return x, adj_mat, y
 
