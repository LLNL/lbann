import unittest 
import os.path 
import sys 
import numpy as np 


root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


sys.path.append(root_dir)

import dataset 



# TO DO: Add data to lustre + gpfs for easier testing 

class dataset_test(unittest.TestCase):
   
    def test_num_train_samples(self):
       #print("Testing num train samples") 
       self.assertEqual(dataset.num_train_samples(), 64)
     
    def test_get_train(self):
         
        #print("Testing get train")
        for i in range(dataset.num_train_samples()):
            mof = dataset.get_train(i)
            self.assertIsInstance(mof,  np.ndarray)
        

    def test_sample_dims(self):
       # print("Testing Sample Dims")
        self.assertEqual(dataset.sample_dims()[0], dataset.get_train(0).size)



if __name__ == '__main__':
    unittest.main()
