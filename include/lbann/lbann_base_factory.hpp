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

#ifndef LBANN_BASE_FACTORY_HPP
#define LBANN_BASE_FACTORY_HPP

#include <map>


using std::shared_ptr;

namespace lbann {

/**
* lbann_base_factory.h  - base factory from which different methods (for layers, optimizer)
* could be registered. This class is to be inherrited by appropriate classes
* e.g., layer_factory, optimizer_factory. Derived classes holds a list of available methods
* of their types e.g., full connected layer, convultional layer for Layer class
*/

template <typename Method>
class base_factory {
 public:
  typedef shared_ptr<Method> method_pointer;
  typedef std::map<pair<string,int>, shared_ptr<Method> > method_container;
  base_factory() {}
  virtual ~base_factory() {}


  void add_method(string sname, shared_ptr<Method> m, int index = 0) {
    std::pair<string,int> name(sname,index);
    if(methods.find(name) != methods.end())
      cerr << "\n LBANN Warning, method list already has a method pointer associated with \""
           << name.first << " at index " << name.second << "\", not added to method list\n";
    else {
      methods[name] = m;
    }
  }

  //remove method()

  shared_ptr<Method> get_method(string sname, int index = 0) {
    std::pair<string,int> name(sname,index);
    return methods[name];
  }

 protected:

  method_container methods;

}; //end base_factory class

} //end lbann namespace

#endif // LBANN_BASE_FACTORY_HPP
