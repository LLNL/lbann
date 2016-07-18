## TBinf - Tensorboard interface

**Author**: Nikoli Dryden <dryden1@llnl.gov>

[Tensorboard](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tensorboard)
is Tensorflow's fancy visualization interface for summary statistics.

This is a light wrapper providing support for writing Tensorboard event files.
Currently, this supports writing scalar and histogram summaries. Support for
image summaries may happen eventually.

To use, create a new instance of `TBinf::SummaryWriter` and provide the
constructor with your desired logging directory. Note that Tensorboard splits
runs up based on subdirectories, so if you wish to do the same, you should
manually provide different subdirectories for each run. You may then use the
`add_scalar` and `add_histogram` methods to create summaries. While optional,
you should probably pass a global step ID when calling these for Tensorboard to
display your data nicely.

Tags may be any string, but Tensorboard will use `/`s to separate variable names
and scope them.

Here is a quick example:

```
#include "TBinf.hpp"

// ...

std::string logdir = "mloutput/testrun1";
TBinf::SummaryWriter sw(logdir);
for (int64_t step = 1; step <= 3; ++step) {
  sw.add_scalar("test/someval", 42.0f, step);
  std::vector<double> histo_vals;
  histo_vals.push_back(-3.0);
  histo_vals.push_back(5.0);
  sw.add_histogram("test/somehist", vals, step);
}
```

## Installation and dependencies

TBinf needs a working installation of [Google Protocol Buffers](https://developers.google.com/protocol-buffers/)
supporting the proto3 syntax and a compiler that supports C++11.

To compile the protobuf messages (`protoc` is the Protocol Buffers compiler):
```
protoc -I=./ --cpp_out=./ summary.proto event.proto
```

To build (you should probably use this as part of a larger project, this does
not build a library (or include a `main`)):
```
CXX -std=c++11 -I$PROTOBUF_INCLUDE -L$PROTOBUF_LIB -lprotobuf TBinf.cpp event.pb.cc summary.pb.cc
```

## License notice

This code contains portions derived from code in the Tensorflow project.
Specifically, `event.proto`, `summary.proto`, and `tbext.hpp` contain modified
code. The original license notification is as follows:

```
Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
