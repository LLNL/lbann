std::string const hdf5_reader = R"ptext(
data_reader {
  reader {
    name: "hdf5_data_reader"
    role: "train"
    shuffle: false
    sample_list: "sample_list_jag.txt"
    percent_of_data_to_use: 1.0

    data_schema_filename: "hdf5_data_schema.yaml"
    experiment_schema_filename: "hdf5_experiment_schema.yaml"
    validation_percent: 0.1
  }
}
)ptext";


