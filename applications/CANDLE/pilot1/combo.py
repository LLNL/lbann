import lbann
import lbann.modules.base

def list2str(l):
    return ' '.join(str(l))

#FC, Activation, Dropout for each track in Combo network
class TrackModule(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    #todo, add kp and activation
    def __init__(self,neuron_dims=[1000,1000,1000],activation=lbann.Relu, keep_prob=0.95, weights=[], name=None):
       self.instance = 0
       self.name = (name if name
                     else 'combo{0}'.format(TrackModule.global_count))
      
       self.kp = keep_prob 
       fc = lbann.modules.FullyConnectedModule
       if(weights):
         self.track_fc = [fc(neuron_dims[i],activation=activation,weights=[weights[2*i],weights[2*i+1]], name=self.name+'fc'+str(i))
                      for i in range(len(neuron_dims))]
       else:
         self.track_fc = [fc(neuron_dims[i],activation=activation, name=self.name+'fc'+str(i))
                      for i in range(len(neuron_dims))]
      
    def forward(self,x):
        return lbann.Dropout(self.track_fc[2](
                     lbann.Dropout(self.track_fc[1](
                     lbann.Dropout(self.track_fc[0](x),keep_prob=self.kp)),keep_prob=self.kp)),keep_prob=self.kp)
 

class Combo(lbann.modules.Module):

    global_count = 0  # Static counter, used for default names

    def __init__(self,neuron_dims=[1000,1000,1000], activation=lbann.Relu, keep_prob=0.95,  name=None):
       self.instance = 0
       self.name = (name if name
                     else 'combo{0}'.format(Combo.global_count))


       #shared weights for drug 1 and 2 tracks 
       shared_w=[]
       for i in range(len(neuron_dims)):
         shared_w.append(lbann.Weights(initializer=lbann.HeNormalInitializer(),
                              name='drug_matrix'+ str(i)))
         shared_w.append(lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0),
                              name='drug_bias'+str(i)))
       print("SHARED W ", type(shared_w))

       self.geneT = TrackModule(neuron_dims, activation, keep_prob, name=self.name+'gene_track')
       self.drug1T = TrackModule(neuron_dims,activation, keep_prob, shared_w, name=self.name+'drug1_track')
       self.drug2T = TrackModule(neuron_dims, activation, keep_prob,shared_w, name = self.name+'drug2_track')
       self.concatT = TrackModule(neuron_dims, activation, keep_prob, name=self.name+'concat_track')
      
    def forward(self,x):
         x_slice = lbann.Slice(x, axis=0, slice_points="0 921 4750 8579",name='inp_slice')
         gene = self.geneT(lbann.Identity(x_slice))
         drug1 = self.drug1T(lbann.Identity(x_slice))
         drug2 = self.drug2T(lbann.Identity(x_slice))
         concat = self.concatT(lbann.Concatenation([gene, drug1, drug2], name=self.name+'concat'))
         response_fc = lbann.FullyConnected(concat,num_neurons = 1, has_bias = True) 
         return response_fc
         
