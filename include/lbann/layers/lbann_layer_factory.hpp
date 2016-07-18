////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC. 
// Produced at the Lawrence Livermore National Laboratory. 
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN. 
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_FACTORY_HPP_INCLUDED
#define LBANN_LAYER_FACTORY_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/lbann_base_factory.hpp"
#include <string>
#include <type_traits>

using namespace std;



namespace lbann
{
/**
* lbann_layer_factory.h -factory to create/hold different type of layers
*  available to a model
*/
class layer_factory: private base_factory<Layer>
{

  private:
    typedef base_factory<Layer> layer_types;

  public:
    typedef typename layer_types::method_pointer layer_pointer;
    typedef typename layer_types::method_container layer_container;


    layer_factory(){}


    ~layer_factory(){}


    //Access methods
  template <class Derived, class... Args>
  Layer* create_layer(string layer_label, int layer_index, Args&&... args)
  {
    Layer* in_ptr = new Derived(layer_index, std::forward<Args>(args)...);
    layer_types::add_method(layer_label, layer_pointer(in_ptr),layer_index);
    return in_ptr;
  }


  layer_pointer get_layer(string layer_label,int layer_index=0)
  {
    layer_pointer lp = layer_types::get_method(layer_label,layer_index);
    if(lp.get() == NULL) {
      exit(-1);
    }
    return lp;
  }

  //void remove_layer(index)

  inline size_t size()
  {
    return layer_types::methods.size();
  }

  //Debug: Layers in my world
  void print() const
  {
    cout << "Layers in my World (label:index): " << endl;
      for(auto& i:layer_types::methods)
        cout << i.first.first << " " << i.first.second << endl;
      //it->second.print()
  }



};//end layer_factory class


}// end lbann namespace


#endif // LBANN_LAYER_FACTORY_HPP_INCLUDED
