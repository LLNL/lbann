var NAVTREE =
[
  [ "LBANN", "index.html", [
    [ "Overview", "index.html", [
      [ "LBANN Development Team", "index.html#team", null ],
      [ "License", "index.html#license", null ]
    ] ],
    [ "Getting Started", "getting_started.html", [
      [ "Download", "getting_started.html#getting_started_download", null ],
      [ "Building LBANN", "getting_started.html#getting_started_building", [
        [ "Livermore Computing Build", "getting_started.html#lc", null ],
        [ "Spack Build", "getting_started.html#spack", null ],
        [ "OSX Build", "getting_started.html#osx", null ],
        [ "Dependency List", "getting_started.html#dependencies", null ]
      ] ],
      [ "Basic Usage", "getting_started.html#getting_started_basicusage", [
        [ "Verification", "getting_started.html#verification", null ]
      ] ]
    ] ],
    [ "Callbacks", "callbacks.html", [
      [ "Check Dataset", "callbacks.html#checkdata", null ],
      [ "Check Initialization", "callbacks.html#checkinit", null ],
      [ "Check Reconstruction", "callbacks.html#checkreconstruction", null ],
      [ "Check NAN", "callbacks.html#checknan", null ],
      [ "Check Small", "callbacks.html#checksmall", null ],
      [ "Dump Activations", "callbacks.html#dump_acts", null ],
      [ "Dump Gradients", "callbacks.html#dump_grads", null ],
      [ "Dump Minibatch Sample Indices", "callbacks.html#dump_mb_sample_indices", null ],
      [ "Dump Weights", "callbacks.html#dump_wei", null ],
      [ "Early Stopping", "callbacks.html#earlystop", null ],
      [ "Gradient Check", "callbacks.html#gradientcheck", null ],
      [ "Hang", "callbacks.html#hang", null ],
      [ "Inter-model Communication", "callbacks.html#im_comm", null ],
      [ "Print IO", "callbacks.html#io", null ],
      [ "Learning Rate", "callbacks.html#learningrate", null ],
      [ "Manage LTFB", "callbacks.html#LTFB", null ],
      [ "Print Accuracy", "callbacks.html#print_acc", null ],
      [ "Save Images", "callbacks.html#save_images", null ],
      [ "Summary", "callbacks.html#summary", null ],
      [ "Timer", "callbacks.html#timer", null ],
      [ "Debug", "callbacks.html#dbg", null ],
      [ "Variable Minibatch", "callbacks.html#variable_mb", null ]
    ] ],
    [ "Layers", "layers.html", [
      [ "Learning", "layers.html#learning", [
        [ "Convolution", "layers.html#conv", null ],
        [ "Deconvolution", "layers.html#deconv", null ],
        [ "Fully Connected", "layers.html#ip", null ]
      ] ],
      [ "Regularizer", "layers.html#regularizer", [
        [ "Batch Normalization", "layers.html#batchNorm", null ],
        [ "Dropout", "layers.html#dropout", null ],
        [ "Selu Dropout", "layers.html#selu_dropout", null ],
        [ "Local Response Norm Layer", "layers.html#local_response_norm_layer", null ]
      ] ],
      [ "Transform", "layers.html#transform", [
        [ "Concatenation", "layers.html#concatenation", null ],
        [ "Noise", "layers.html#noise", null ],
        [ "Unpooling", "layers.html#unpooling", null ],
        [ "Pooling", "layers.html#pooling", null ],
        [ "Reshape", "layers.html#reshape", null ],
        [ "Slice", "layers.html#slice", null ],
        [ "Split", "layers.html#split", null ],
        [ "Sum", "layers.html#sum", null ]
      ] ],
      [ "Activation", "layers.html#activation", [
        [ "Identity", "layers.html#idlayer", null ],
        [ "Rectified Linear Unit", "layers.html#reluLayer", null ],
        [ "Leaky Relu", "layers.html#leakyrelu", null ],
        [ "Smooth Relu", "layers.html#smoothrelu", null ],
        [ "Exponential Linear Unit", "layers.html#expLinUn", null ],
        [ "Scaled Elu", "layers.html#seluLayer", null ],
        [ "Sigmoid", "layers.html#sigLayer", null ],
        [ "Softplus", "layers.html#softplus", null ],
        [ "Softmax", "layers.html#softmax", null ],
        [ "Tanh", "layers.html#tanh", null ],
        [ "Atan", "layers.html#atan", null ],
        [ "Bent Identity", "layers.html#bent_identity", null ],
        [ "Exponential", "layers.html#exponential", null ]
      ] ],
      [ "IO", "layers.html#i_o", [
        [ "Input", "layers.html#input", null ],
        [ "Target", "layers.html#target", null ]
      ] ]
    ] ],
    [ "Metrics", "metrics.html", [
      [ "Categorical Accuracy", "metrics.html#cataccuracy", null ],
      [ "Mean Absolute Deviation", "metrics.html#mean_abs_dev", null ],
      [ "Mean Squared Error", "metrics.html#mse", null ],
      [ "Pearson Correlation", "metrics.html#pearson", null ],
      [ "Top K Categorical Accuracy", "metrics.html#top_k", null ]
    ] ],
    [ "Objective Functions", "obj_fn.html", [
      [ "Loss Functions", "obj_fn.html#loss_functions", [
        [ "Binary Cross Entropy", "obj_fn.html#bin_cross_ent", null ],
        [ "Cross Entropy", "obj_fn.html#cross_ent", null ],
        [ "Cross Entropy with Uncertainty", "obj_fn.html#cross_ent_uncertain", null ],
        [ "Geometric Negative Log Likelihood", "obj_fn.html#gemo_negloglike", null ],
        [ "Mean Absolute Deviation", "obj_fn.html#mad", null ],
        [ "Mean Squared Error", "obj_fn.html#m_s_e", null ],
        [ "Poisson Negative Log Likelihood", "obj_fn.html#pos_negloglike", null ],
        [ "Polya Negative Log Likelihood", "obj_fn.html#poly_negloglike", null ]
      ] ],
      [ "Weight Regularization", "obj_fn.html#weight_regularization", [
        [ "L2 Weight Regularization", "obj_fn.html#l2_weight", null ]
      ] ]
    ] ],
    [ "Optimizers", "optimizers.html", [
      [ "Adagrad", "optimizers.html#Adagrad", null ],
      [ "Adam", "optimizers.html#Adam", null ],
      [ "Hypergradient Adam", "optimizers.html#hadam", null ],
      [ "RMSprop", "optimizers.html#rmsp", null ],
      [ "SGD", "optimizers.html#SGD", null ]
    ] ],
    [ "Todo List", "todo.html", null ],
    [ "Modules", "modules.html", "modules" ],
    [ "Namespaces", null, [
      [ "Namespace List", "namespaces.html", "namespaces" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Typedefs", "functions_type.html", null ],
        [ "Enumerations", "functions_enum.html", null ],
        [ "Enumerator", "functions_eval.html", null ],
        [ "Related Functions", "functions_rela.html", null ]
      ] ]
    ] ],
    [ "Files", null, [
      [ "File List", "files.html", "files" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Functions", "globals_func.html", null ],
        [ "Variables", "globals_vars.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Enumerations", "globals_enum.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"Elemental__extensions_8cpp.html",
"callbacks.html#io",
"classlbann_1_1adam.html#aca07e925a4751fc5c10a62fb9c72c896",
"classlbann_1_1csv__reader.html#a3a6cb698fbeb4abd0032e2b018bbb326",
"classlbann_1_1data__store__csv.html#a8920e09beb3e40e59e2858daf95a6cf9",
"classlbann_1_1fully__connected__layer.html#a1f0ae7a5f172e5ced011eaa2755fd04e",
"classlbann_1_1generic__input__layer.html#a0f09422a09e76da77ac42b3037173041",
"classlbann_1_1image__data__reader.html#ae3f4a0b018e8212a42cbbfbd2b514bf4",
"classlbann_1_1lbann__callback__check__reconstruction__error.html#ad277f418a2c28db941ab44398ee8be5e",
"classlbann_1_1lbann__callback__optimizerwise__adaptive__learning__rate.html#acf561f3ca522caa6f9d8e8333e5138cc",
"classlbann_1_1lbann__comm.html#a63eae73674cbadea2eff8b6595322336",
"classlbann_1_1loss__function.html#a17a3629e8519c4215ad776a0c7b522ae",
"classlbann_1_1numpy__reader.html#a5e0d81c07c950a2d81645566a2ccd965",
"classlbann_1_1polya__negloglike.html#a0aec768ffa50a8b716149d597699eddb",
"classlbann_1_1slice__layer.html#a8c2b93a3ed24b239cc429ee85ee7ef29",
"cnpy__utils_8cpp.html#ab19ad0a361570b7e78e203c02d6ba13a",
"dir_165ca8a5737c568fa66c431fe2819b03.html",
"graph_8cpp.html#aeb19a22d8fac402df104ed8d547a10ee",
"obj_fn.html#poly_negloglike",
"selu__dropout_8hpp.html",
"weights__factory_8cpp.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';