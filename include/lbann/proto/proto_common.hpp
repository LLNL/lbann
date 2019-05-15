#ifndef LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED
#define LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED

#include "lbann/lbann.hpp"
#include <lbann.pb.h>
#include "lbann/proto/factories.hpp"

namespace lbann {

/** @brief Returns true if the Model contains at least one MotifLayer */
bool has_motifs(const lbann_comm& comm, const lbann_data::LbannPB& p);

void expand_motifs(const lbann_comm& comm, lbann_data::LbannPB& pb);

/** @brief Customize the name of the index list
 *
 *  The following options are available
 *   - trainer ID
 *   - model name
 *
 *  The format for the naming convention if the provided name is
 *  \<index list\> is:
 *  @verbatim
    <index list> == <basename>.<extension>
    <model name>_t<ID>_<basename>.<extension> @endverbatim
 */
void customize_data_readers_index_list(const lbann_comm& comm,
                                       lbann_data::LbannPB& p);

/** @brief instantiates one or more generic_data_readers and inserts
 *         them in &data_readers
 */
void init_data_readers(
  lbann_comm *comm,
  const lbann_data::LbannPB& p,
  std::map<execution_mode, generic_data_reader *>& data_readers,
  bool is_shareable_training_data_reader,
  bool is_shareable_testing_data_reader,
  bool is_shareable_validation_data_reader = false);

/** @brief adjusts the number of parallel data readers */
void set_num_parallel_readers(const lbann_comm& comm, lbann_data::LbannPB& p);

/** @brief adjusts the values in p by querying the options db */
void get_cmdline_overrides(const lbann_comm& comm, lbann_data::LbannPB& p);

/** @brief print various params (learn_rate, etc) to cout */
void print_parameters(const lbann_comm& comm, lbann_data::LbannPB& p);

/** @brief prints usage information */
void print_help(const lbann_comm& comm);

/** @brief prints usage information */
void print_help(std::ostream& os);

/** @brief prints prototext file, cmd line, etc to file */
void save_session(const lbann_comm& comm,
                  const int argc, char * const* argv,
                  lbann_data::LbannPB& p);

/** @brief Read prototext from a file into a protobuf message. */
void read_prototext_file(
  const std::string& fn,
  lbann_data::LbannPB& pb,
  const bool master);

/** @brief Write a protobuf message into a prototext file. */
bool write_prototext_file(
  const std::string& fn,
  lbann_data::LbannPB& pb);

} // namespace lbann

#endif // LBANN_PROTO_PROTO_COMMON_HPP_INCLUDED
