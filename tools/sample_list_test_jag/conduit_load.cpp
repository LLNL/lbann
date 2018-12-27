#include <iostream>
#include <cstdio>
#include <string>
#include <set>
#include <vector>
#include <map>
#include "data_reader_jag_conduit.hpp"
#include "lbann/utils/glob.hpp"
#include <chrono>
#include <cstdio>
#include <algorithm>

// The number of channels in a JAG image
const int num_channels = 4;
using variable_t = lbann::data_reader_jag_conduit::variable_t;

struct stats_t {
  std::vector<double> m_sum;
  std::vector<double> m_min;
  std::vector<double> m_max;
  size_t m_num_bins;
  std::vector< std::vector<size_t> > m_histo;

  bool set_num_bins(size_t n) {
    m_num_bins = n;
    return (n > 0u);
  }

  void init(size_t num_variables) {
    m_sum.assign(num_variables, 0.0);
    m_min.assign(num_variables, std::numeric_limits<double>::max());
    m_max.assign(num_variables, std::numeric_limits<double>::lowest());
    std::vector<size_t> empty_bins(m_num_bins, 0u);
    m_histo.assign(num_variables, empty_bins);
  }

  void add_to_bin(size_t idx_var, double value) {
    int bin = static_cast<int>(value * m_num_bins);
    bin = std::min(bin, static_cast<int>(m_num_bins-1));
    bin = std::max(bin, 0);
    (m_histo.at(idx_var))[bin] ++;
  }
};

inline double get_time() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(
           steady_clock::now().time_since_epoch()).count();
}

std::string to_string(const variable_t vt);

void show_help(int argc, char** argv);

void configuration(lbann::data_reader_jag_conduit& dreader, const bool use_all, const bool do_normalize, variable_t response_t);

template <typename T>
std::ostream& print(const std::vector<T>& keys, std::ostream& os);
void print_keys(const std::vector<std::string>& keys);

void show_details(const lbann::data_reader_jag_conduit& dreader);

void fetch_iteration(const bool wrt_response,
                     const bool stat_response,
                     const size_t mb_size,
                     lbann::data_reader_jag_conduit& dreader,
                     stats_t& response_stats);

void report_summary_stats(const lbann::data_reader_jag_conduit& dreader,
                     stats_t& response_stats);


int main(int argc, char** argv)
{

  if (argc != 10) {
    show_help(argc, argv);
    return 0;
  }

  const std::string file_path_pattern = std::string(argv[1]);
  std::vector<std::string> filenames = lbann::glob(file_path_pattern);
  if (filenames.size() < 1u) {
    std::cerr << "Failed to get data filenames from: " + file_path_pattern << std::endl;
    return 0;
  }


  bool use_all = (atoi(argv[2]) > 0);
  bool write_schema = (atoi(argv[3]) > 0);
  bool write_images = (atoi(argv[4]) > 0);
  bool measure_time = (atoi(argv[5]) > 0);
  bool do_normalize = (atoi(argv[6]) > 0);
  bool wrt_response = (atoi(argv[7]) > 0);
  bool stat_response = (atoi(argv[8]) > 0);
  variable_t response_type = static_cast<variable_t>(atoi(argv[9]));

  std::cout << "use_all_vars : " << use_all << std::endl;
  std::cout << "write_schema : " << write_schema << std::endl;
  std::cout << "write_images : " << write_images << std::endl;
  std::cout << "measure_time : " << measure_time << std::endl;
  std::cout << "do_normalize : " << do_normalize << std::endl;
  std::cout << "wrt_response : " << wrt_response << std::endl;
  std::cout << "stat_response : " << stat_response << std::endl;
  std::cout << "response_type : " << to_string(response_type) << std::endl;
  if (stat_response && wrt_response) {
    std::cerr << "Both wrt_response and stat_response cannot be set simultaneously" << std::endl;
    return 0;
  }

  using namespace lbann;

  std::shared_ptr<cv_process> pp;
  pp = std::make_shared<cv_process>();

#ifdef LBANN_HAS_CONDUIT
  using namespace std;
  data_reader_jag_conduit dreader(pp);

  configuration(dreader, use_all, do_normalize, response_type);

  double t_load =  get_time();
  for (const auto& data_file: filenames) {
    std::cout << "loading " << data_file << std::endl;
    size_t num_valid_samples = 0ul;

    dreader.load_conduit(data_file, num_valid_samples);
  }
  std::cout << "time to load consuit file: " << get_time() - t_load << " (sec)" << std::endl;

  dreader.check_image_data();

  show_details(dreader);

  const size_t n = dreader.get_num_valid_local_samples();
  // prepare the fake base data reader class (generic_data_reader) for mini batch data accesses
  const size_t mb_size = 64u;
  dreader.set_num_samples(n);
  dreader.set_mini_batch_size(mb_size);
  dreader.init();
  dreader.check_image_data();

  if (measure_time) {
    double t =  get_time();
    stats_t response_stats;
    response_stats.set_num_bins(10u);

    fetch_iteration(wrt_response, stat_response, mb_size, dreader, response_stats);

    std::cout << "time to read all the samples: " << get_time() - t << " (sec)" << std::endl;

    if (stat_response) {
      std::cout << "Statistics of the response variables:" << std::endl;
      report_summary_stats(dreader, response_stats);
    }
  } else if (stat_response) {
    stats_t response_stats;
    response_stats.set_num_bins(10u);

    fetch_iteration(wrt_response, stat_response, mb_size, dreader, response_stats);

    std::cout << "Statistics of the response variables:" << std::endl;
    report_summary_stats(dreader, response_stats);
  }

  if (write_schema) { // print schemas
    for (size_t s = 0u; s < n; ++s) {
      std::ofstream schema_out("schema." + std::to_string(s) + ".txt");
      std::streambuf *org_buf = std::cout.rdbuf(); // save old buf
      std::cout.rdbuf(schema_out.rdbuf()); // redirect std::cout to schema_out
      dreader.print_schema(s);
      std::cout.rdbuf(org_buf);
    }
  }

  if (write_images) { // dump image files
    for (size_t s = 0u; s < n; ++s) {
      std::vector<cv::Mat> images = dreader.get_cv_images(s);
      for (size_t i = 0u; i < images.size(); ++i) {
        if (dreader.check_split_image_channels()) {
          cv::normalize(images[i], images[i], 0, 256, cv::NORM_MINMAX);
          cv::imwrite("image." + std::to_string(s) + '.' + std::to_string(i/num_channels)
                      + '.' + std::to_string(i%num_channels) + ".png", images[i]);
        } else {
          cv::Mat ch[num_channels];
          cv::split(images[i], ch);
          for(int c=0; c < num_channels; ++c) {
            cv::normalize(ch[c], ch[c], 0, 256, cv::NORM_MINMAX);
            cv::imwrite("image." + std::to_string(s) + '.' + std::to_string(i)
                        + '.' + std::to_string(c) + ".png", ch[c]);
          }
        }
      }
    }
  }

/*
  dreader.get_scalars(0);
  double t =  get_time();
  for (size_t s = 0u; s < n; ++s) {
    dreader.get_inputs(s);
  }
  std::cout << "time to load input data: " << get_time() - t << std::endl;
*/
#endif // LBANN_HAS_CONDUIT

  return 0;
}



void show_help(int argc, char** argv) {
  std::cout << "Uasge: > " << argv[0] << " conduit_file use_all_vars write_schema write_images measure_time do_normalize wrt_response stat_response response_type" << std::endl;
  std::cout << "         - use_all_vars (1|0): whether to use all the variables or only those selected." << std::endl;
  std::cout << "         - write_schema (1|0): whether to write schema files." << std::endl;
  std::cout << "         - write_images (1|0): whether to write image files." << std::endl;
  std::cout << "         - measure_time (1|0): whether to measure the time to read all the samples." << std::endl;
  std::cout << "         - do_normalize (1|0): whether to use normalization." << std::endl;
  std::cout << "         - wrt_response (1|0): whether to print out response variables." << std::endl;
  std::cout << "         - stat_response (1|0): whether to summarize the statistics of response variables." << std::endl;
  std::cout << "           Both wrt_response and stat_response cannot be set simultaneously" << std::endl;
  std::cout << "         - response_type (2|3): type of the response variable: 2 for JAG_Scalar, and 3 for JAG_Input." << std::endl;
}


void set_normalization(lbann::data_reader_jag_conduit& dreader) {

  const auto& image_keys = dreader.get_image_choices();
  const auto& scalar_keys = dreader.get_scalar_choices();
  const auto& input_keys = dreader.get_input_choices();

  using linear_transform_t = lbann::data_reader_jag_conduit::linear_transform_t;

  dreader.clear_image_normalization_params();
  dreader.clear_scalar_normalization_params();
  dreader.clear_input_normalization_params();

  for (const auto& key: image_keys) { // TODO: use unique set of values per key
    std::cout << " - define image normalization for " << key << std::endl;
    dreader.add_image_normalization_param(linear_transform_t(28.128928461, 0.0)); // ch0
    dreader.add_image_normalization_param(linear_transform_t(817.362315273, 0.0)); // ch1
    dreader.add_image_normalization_param(linear_transform_t(93066.843470244, 0.0)); // ch2
    dreader.add_image_normalization_param(linear_transform_t(4360735.362407147, 0.0)); // ch3
  }

  for (const auto& key: scalar_keys) {
    std::cout << " - define scalar normalization for " << key << std::endl;
  #if 0 // 10K
    if (key == "BWx")
      dreader.add_scalar_normalization_param(linear_transform_t(1.660399380e+01, -8.914478521e-01)); // BWx
    else if (key == "BT")
      dreader.add_scalar_normalization_param(linear_transform_t(1.499062171e+00, -3.529513015e+00)); // BT
    else if (key == "tMAXt")
      dreader.add_scalar_normalization_param(linear_transform_t(1.530702521e+00, -3.599429878e+00)); // tMAXt
    else if (key == "BWn")
      dreader.add_scalar_normalization_param(linear_transform_t(4.644040100e+01, -1.703187287e+00)); // BWn
    else if (key == "MAXpressure")
      dreader.add_scalar_normalization_param(linear_transform_t(1.795164343e-06, -5.849243445e-01)); // MAXpressure
    else if (key == "BAte")
      dreader.add_scalar_normalization_param(linear_transform_t(2.807222136e-01, -1.042499360e+00)); // BAte
    else if (key == "MAXtion")
      dreader.add_scalar_normalization_param(linear_transform_t(2.571981124e-01, -1.050431705e+00)); // MAXtion
    else if (key == "tMAXpressure")
      dreader.add_scalar_normalization_param(linear_transform_t(1.468048973e+00, -3.447884539e+00)); // tMAXpressure
    else if (key == "BAt")
      dreader.add_scalar_normalization_param(linear_transform_t(2.807222136e-01, -1.042499360e+00)); // BAt
    else if (key == "Yn")
      dreader.add_scalar_normalization_param(linear_transform_t(8.210767783e-18, -2.182660862e-02)); // Yn
    else if (key == "Ye")
      dreader.add_scalar_normalization_param(linear_transform_t(3.634574711e-03, -2.182660596e-02)); // Ye
    else if (key == "Yx")
      dreader.add_scalar_normalization_param(linear_transform_t(2.242376030e-02, -3.376249820e-01)); // Yx
    else if (key == "tMAXte")
      dreader.add_scalar_normalization_param(linear_transform_t(1.530702521e+00, -3.599429878e+00)); // tMAXte
    else if (key == "BAtion")
      dreader.add_scalar_normalization_param(linear_transform_t(2.807222136e-01, -1.042499360e+00)); // BAtion
    else if (key == "MAXte")
      dreader.add_scalar_normalization_param(linear_transform_t(2.571981124e-01, -1.050431705e+00)); // MAXte
    else if (key == "tMAXtion")
      dreader.add_scalar_normalization_param(linear_transform_t(1.530702521e+00, -3.599429878e+00)); // tMAXtion
    else if (key == "BTx")
      dreader.add_scalar_normalization_param(linear_transform_t(1.461374463e+00, -3.414620490e+00)); // BTx
    else if (key == "MAXt")
      dreader.add_scalar_normalization_param(linear_transform_t(2.571981124e-01, -1.050431705e+00)); // MAXt
    else if (key == "BTn")
      dreader.add_scalar_normalization_param(linear_transform_t(1.499062171e+00, -3.529513015e+00)); // BTn
    else if (key == "BApressure")
      dreader.add_scalar_normalization_param(linear_transform_t(2.240009139e-06, -5.837354616e-01)); // BApressure
    else if (key == "tMINradius")
      dreader.add_scalar_normalization_param(linear_transform_t(1.427286973e+00, -3.328267524e+00)); // tMINradius
    else if (key == "MINradius")
      dreader.add_scalar_normalization_param(linear_transform_t(6.404465614e-02, -1.418863592e+00)); // MINradius
  #else
    if (key == "BWx")
      dreader.add_scalar_normalization_param(linear_transform_t(1.5420795e+01, -8.313582e-01)); // BWx
    else if (key == "BT")
      dreader.add_scalar_normalization_param(linear_transform_t(1.4593427e+00, -3.426026e+00)); // BT
    else if (key == "tMAXt")
      dreader.add_scalar_normalization_param(linear_transform_t(1.4901131e+00, -3.493689e+00)); // tMAXt
    else if (key == "BWn")
      dreader.add_scalar_normalization_param(linear_transform_t(4.4250137e+01, -1.623055e+00)); // BWn
    else if (key == "MAXpressure")
      dreader.add_scalar_normalization_param(linear_transform_t(2.4432852e-06, -7.724349e-01)); // MAXpressure
    else if (key == "BAte")
      dreader.add_scalar_normalization_param(linear_transform_t(2.6368040e-01, -9.765773e-01)); // BAte
    else if (key == "MAXtion")
      dreader.add_scalar_normalization_param(linear_transform_t(2.4198603e-01, -9.856284e-01)); // MAXtion
    else if (key == "tMAXpressure")
      dreader.add_scalar_normalization_param(linear_transform_t(1.4302059e+00, -3.349900e+00)); // tMAXpressure
    else if (key == "BAt")
      dreader.add_scalar_normalization_param(linear_transform_t(2.6368040e-01, -9.765773e-01)); // BAt
    else if (key == "Yn")
      dreader.add_scalar_normalization_param(linear_transform_t(7.1544386e-18, -1.869906e-02)); // Yn
    else if (key == "Ye")
      dreader.add_scalar_normalization_param(linear_transform_t(3.1669860e-03, -1.869906e-02)); // Ye
    else if (key == "Yx")
      dreader.add_scalar_normalization_param(linear_transform_t(2.1041247e-02, -3.084058e-01)); // Yx
    else if (key == "tMAXte")
      dreader.add_scalar_normalization_param(linear_transform_t(1.4901131e+00, -3.493689e+00)); // tMAXte
    else if (key == "BAtion")
      dreader.add_scalar_normalization_param(linear_transform_t(2.6368040e-01, -9.765773e-01)); // BAtion
    else if (key == "MAXte")
      dreader.add_scalar_normalization_param(linear_transform_t(2.4198603e-01, -9.856284e-01)); // MAXte
    else if (key == "tMAXtion")
      dreader.add_scalar_normalization_param(linear_transform_t(1.4901131e+00, -3.493689e+00)); // tMAXtion
    else if (key == "BTx")
      dreader.add_scalar_normalization_param(linear_transform_t(1.3456596e+00, -3.116023e+00)); // BTx
    else if (key == "MAXt")
      dreader.add_scalar_normalization_param(linear_transform_t(2.4198603e-01, -9.856284e-01)); // MAXt
    else if (key == "BTn")
      dreader.add_scalar_normalization_param(linear_transform_t(1.4593427e+00, -3.426026e+00)); // BTn
    else if (key == "BApressure")
      dreader.add_scalar_normalization_param(linear_transform_t(3.0520000e-06, -7.714907e-01)); // BApressure
    else if (key == "tMINradius")
      dreader.add_scalar_normalization_param(linear_transform_t(1.3925443e+00, -3.239921e+00)); // tMINradius
    else if (key == "MINradius")
      dreader.add_scalar_normalization_param(linear_transform_t(1.0023756e-01, -2.815272e+00)); // MINradius
  #endif
  }

  for (const auto& key: input_keys) {
    std::cout << " - define input normalization for " << key << std::endl;
  #if 0
    if (key == "shape_model_initial_modes:(4,3)")
      dreader.add_input_normalization_param(linear_transform_t(1.667587753e+00, 4.997824968e-01)); // shape_model_initial_modes:(4,3)
    else if (key == "betti_prl15_trans_u")
      dreader.add_input_normalization_param(linear_transform_t(1.000245480e+00, -8.438836401e-05)); // betti_prl15_trans_u
    else if (key == "betti_prl15_trans_v")
      dreader.add_input_normalization_param(linear_transform_t(1.000870539e+00, -7.346414236e-04)); // betti_prl15_trans_v
    else if (key == "shape_model_initial_modes:(2,1)")
      dreader.add_input_normalization_param(linear_transform_t(1.668835219e+00, 4.997744013e-01)); // shape_model_initial_modes:(2,1)
    else if (key == "shape_model_initial_modes:(1,0)")
      dreader.add_input_normalization_param(linear_transform_t(1.667992865e+00, 4.999102733e-01)); // shape_model_initial_modes:(1,0)
  #else
    if (key == "shape_model_initial_modes:(4,3)")
      dreader.add_input_normalization_param(linear_transform_t(1.666721654e+01, 5.000145788e+00)); // shape_model_initial_modes:(4,3)
    else if (key == "betti_prl15_trans_u")
      dreader.add_input_normalization_param(linear_transform_t(1.000025133e+01, -1.603520955e-06)); // betti_prl15_trans_u
    else if (key == "betti_prl15_trans_v")
      dreader.add_input_normalization_param(linear_transform_t(1.000001645e+00, -1.406676728e-06)); // betti_prl15_trans_v
    else if (key == "shape_model_initial_modes:(2,1)")
      dreader.add_input_normalization_param(linear_transform_t(1.666672975e+00, 4.999989818e-01)); // shape_model_initial_modes:(2,1)
    else if (key == "shape_model_initial_modes:(1,0)")
      dreader.add_input_normalization_param(linear_transform_t(1.666668753e+00, 5.000004967e-01)); // shape_model_initial_modes:(1,0)
  #endif
  }
}


void configuration(lbann::data_reader_jag_conduit& dreader,
  const bool use_all_vars,
  const bool do_normalize,
  const variable_t response_t) {

  using namespace lbann;

  if (!((response_t == data_reader_jag_conduit::JAG_Image) ||
      (response_t == data_reader_jag_conduit::JAG_Scalar) ||
      (response_t == data_reader_jag_conduit::JAG_Input))) {
    std::cerr << "unknown response type: " << static_cast<int>(response_t) << std::endl;
    exit(0);
  }

  const std::vector< std::vector<variable_t> > independent
    = {{data_reader_jag_conduit::JAG_Image, data_reader_jag_conduit::JAG_Scalar}, {data_reader_jag_conduit::JAG_Input} };
  dreader.set_independent_variable_type(independent);

  const std::vector< std::vector<variable_t> > dependent = {{response_t}};
  dreader.set_dependent_variable_type(dependent);

  const std::vector<std::string> image_keys_manual = {
    "(0.0, 0.0)/0.0",
    "(90.0, 0.0)/-0.01",
    "(90.0, 78.0)/-0.03"
  };

  const std::vector<std::string> scalar_keys_manual = {
    "BWx",
    "BT",
    "tMAXt",
    "BWn",
    "MAXpressure",
    "BAte",
    "MAXtion",
    "tMAXpressure",
    "BAt",
    "Yn",
    "Ye",
    "Yx",
    "tMAXte",
    "BAtion",
    "MAXte",
    "tMAXtion",
    "BTx",
    "MAXt",
    "BTn",
    "BApressure",
    "tMINradius",
    "MINradius"
  };

  const std::vector<std::string> input_keys_manual = {
    "shape_model_initial_modes:(4,3)",
    "betti_prl15_trans_u",
    "betti_prl15_trans_v",
    "shape_model_initial_modes:(2,1)",
    "shape_model_initial_modes:(1,0)"
  };

  if (use_all_vars) {
    dreader.set_all_scalar_choices();
    dreader.set_all_input_choices();
    dreader.add_scalar_prefix_filter(std::make_pair("image_(", 26));
    dreader.add_scalar_filter("iBT");
  } else {
    dreader.set_scalar_choices(scalar_keys_manual);
    dreader.set_input_choices(input_keys_manual);
  }

  dreader.set_image_dims(64, 64, num_channels);
  dreader.set_image_choices(image_keys_manual);
  dreader.set_split_image_channels();

  if (do_normalize) {
    set_normalization(dreader);
  }
}

std::string to_string(const variable_t vt) {
  using namespace lbann;
  switch(vt) {
    case data_reader_jag_conduit::JAG_Image:
      return "Jag_Image";
      break;
    case data_reader_jag_conduit::JAG_Scalar:
      return "JAG_Scalar";
      break;
    case data_reader_jag_conduit::JAG_Input:
      return "JAG_Input";
      break;
    default: return "Undefined";
  }
  return "Undefined";
}

template <typename T>
std::ostream& print(const std::vector<T>& keys, std::ostream& os) {
  for(const auto& key: keys) {
    os << ' ' << key;
  }
  return (os << ' ');
}

void print_keys(const std::vector<std::string>& keys) {
  std::cout << "==========================" << std::endl;
  print(keys, std::cout);
  std::cout << std::endl;
  std::cout << "==========================" << std::endl;
}

void show_details(const lbann::data_reader_jag_conduit& dreader) {
  std::cout << "- number_of_samples: " << dreader.get_num_valid_local_samples() << std::endl;
  std::cout << "- linearized image size: " << dreader.get_linearized_image_size() << std::endl;
  std::cout << "- linearized scalar size: " << dreader.get_linearized_scalar_size() << std::endl;
  std::cout << "- linearized input size: " << dreader.get_linearized_input_size() << std::endl;
  std::cout << "- linearized data size: " << dreader.get_linearized_data_size() << std::endl;
  std::cout << "- linearized response size: " << dreader.get_linearized_response_size() << std::endl;

  std::cout << "- linearized data sizes:";
  for (const auto& sz: dreader.get_linearized_data_sizes()) {
    std::cout << ' ' << sz;
  }
  std::cout << std::endl;

  std::cout << "- slice points for independent vars: ";
  print(dreader.get_slice_points_independent(), std::cout);
  std::cout << std::endl;
  std::cout << "- slice points for dependent vars: " ;
  print(dreader.get_slice_points_dependent(), std::cout);
  std::cout << std::endl;

  print_keys(dreader.get_scalar_choices());
  print_keys(dreader.get_input_choices());
  std::cout << dreader.get_description() << std::endl;
}


void fetch_iteration(const bool wrt_response, const bool stat_response,
                     const size_t mb_size,
                     lbann::data_reader_jag_conduit& dreader,
                     stats_t& stats) {

  const size_t n = dreader.get_num_valid_local_samples();
  const size_t nd = dreader.get_linearized_data_size();
  const size_t nr = dreader.get_linearized_response_size();
  const size_t n_full = (n / mb_size) * mb_size; // total number of samples in full mini batches
  const size_t n_rem  = n - n_full; // number of samples in the last partial mini batch (0 if the last one is full)

  CPUMat X;
  CPUMat Y;

  X.Resize(nd, mb_size);
  Y.Resize(nr, mb_size);

  stats.init(nr);

  if (wrt_response) {
    for (size_t s = 0u; s < n_full; s += mb_size) {
      std::cout << "samples [" << s << ' ' << s+mb_size << ")" << std::endl;
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);

      for (size_t c = 0; c < mb_size; ++c) {
        for(size_t r = 0; r < nr ; ++r) {
          printf("\t%e", Y(r, c));
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;

      dreader.update();
    }
    if (n_rem > 0u) {
      X.Resize(X.Height(), n_rem);
      Y.Resize(Y.Height(), n_rem);
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);

      for (size_t c = 0; c < n_rem; ++c) {
        for(size_t r = 0; r < nr ; ++r) {
          printf("\t%e", Y(r, c));
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;

      dreader.update();
    }
  } else if (stat_response) {
    for (size_t s = 0u; s < n_full; s += mb_size) {
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);

      for (size_t c = 0; c < mb_size; ++c) {
        for(size_t r = 0; r < nr ; ++r) {
          const double val = Y(r,c);
          if (val < stats.m_min[r]) {
            stats.m_min[r] = val;
          }
          if (val > stats.m_max[r]) {
            stats.m_max[r] = val;
          }
          
          stats.m_sum[r] += val;
          stats.add_to_bin(r, val);
        }
      }

      dreader.update();
    }
    if (n_rem > 0u) {
      X.Resize(X.Height(), n_rem);
      Y.Resize(Y.Height(), n_rem);
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);

      for (size_t c = 0; c < n_rem; ++c) {
        for(size_t r = 0; r < nr ; ++r) {
          const double val = Y(r,c);
          if (val < stats.m_min[r]) {
            stats.m_min[r] = val;
          }
          if (val > stats.m_max[r]) {
            stats.m_max[r] = val;
          }
          stats.m_sum[r] += val;
          stats.add_to_bin(r, val);
        }
      }

      dreader.update();
    }
  } else {
    for (size_t s = 0u; s < n_full; s += mb_size) {
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);
      dreader.update();
    }
    if (n_rem > 0u) {
      X.Resize(X.Height(), n_rem);
      Y.Resize(Y.Height(), n_rem);
      dreader.fetch_data(X);
      dreader.fetch_responses(Y);
      dreader.update();
    }
  }
}


void report_summary_stats(const  lbann::data_reader_jag_conduit& dreader,
                          stats_t& stats) {

  const size_t n = dreader.get_num_valid_local_samples();
  const auto response_types = dreader.get_dependent_variable_type();

  if (response_types.size() != 1u) {
    std::cerr << "Not able to handle composite responses. Use only one type." << std::endl;
    return;
  }

  const auto& image_keys = dreader.get_image_choices();
  const auto& scalar_keys = dreader.get_scalar_choices();
  const auto& input_keys = dreader.get_input_choices();

  using namespace lbann;
  using variable_t = lbann::data_reader_jag_conduit::variable_t;

  const std::map<variable_t, const std::vector<std::string>*> keys = {
    std::make_pair(data_reader_jag_conduit::JAG_Image, &image_keys),
    std::make_pair(data_reader_jag_conduit::JAG_Scalar, &scalar_keys),
    std::make_pair(data_reader_jag_conduit::JAG_Input, &input_keys),
  };

  const auto it_keys = keys.find(response_types[0]);
  if (it_keys == keys.cend() || (it_keys->second == nullptr)) {
    std::cerr << "Unknown dependent variable keys." << std::endl;
    return;
  }
  const std::vector<std::string> response_keys = *(it_keys->second);

  const size_t nr = dreader.get_linearized_response_size();

  std::cout << "min: ";
  for (size_t r = 0u; r < nr; ++r) {
    std::cout << '\t' << stats.m_min[r];
  }
  std::cout << std::endl;
  std::cout << "max: ";
  for (size_t r = 0u; r < nr; ++r) {
    std::cout << '\t' << stats.m_max[r];
  }
  std::cout << std::endl;
  std::cout << "mean: ";
  for (size_t r = 0u; r < nr; ++r) {
    std::cout << '\t' << stats.m_sum[r]/n;
  }
  std::cout << std::endl;
  std::cout << "histo: \n";
  for (size_t r = 0u; r < nr; ++r) {
    for (size_t b = 0u; b < stats.m_num_bins; ++b) {
      std::cout << '\t' << stats.m_histo[r][b];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "linear transform parameters: " << std::endl;
  std::vector<double> alpha(nr);
  std::vector<double> beta(nr);

  for (size_t r = 0u; r < nr; ++r) {
    alpha[r] = 1.0/(stats.m_max[r] - stats.m_min[r]);
    beta[r] = -stats.m_min[r]/(stats.m_max[r] - stats.m_min[r]);
    const char comma = (r + 1u == nr)? ' '  : ',';
    printf("      { scale: %.9e \tbias: %.9e}%c \t# %s\n", alpha[r], beta[r], comma, response_keys[r].c_str());
  }
  std::cout << std::endl;
}
