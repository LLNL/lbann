import os 
import os.path 
import sys

import unittest 
import numpy as np 



from data.MNIST_Superpixel.dataset import MNIST_Superpixel_Dataset

#import dataset as reader

class TestMSDatabse(unittest.TestCase):
    def setUp(self):
        self.dataset = MNIST_Superpixel_Dataset(train=True, processed=True)
    
    def test_length(self):
        self.assertEqual(len(self.dataset ), 60000)

    def test_dims(self):
        element = self.dataset[0] 
        self.assertEqual(len(element), (75 + (75*75)+10))
    #def test_single_element(self): #Test the shape and structure of the first element 
    #def test_all_elements(self): #Test the shape and structure of all element 


if __name__ == '__main__':
    unittest.main()

