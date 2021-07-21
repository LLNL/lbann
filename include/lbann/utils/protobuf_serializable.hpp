#ifndef LBANN_INCLUDE_LBANN_UTILS_PROTOBUF_SERIALIZABLE_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_UTILS_PROTOBUF_SERIALIZABLE_HPP_INCLUDED

#include <google/protobuf/message.h>

namespace lbann {

/** @brief Represents a class that is describable in LBANN's protobuf
 *         specification.
*/
class ProtobufSerializable
{
public:
  virtual ~ProtobufSerializable() = default;
  /** @brief Write the object to a protobuf message. */
  virtual void write_proto(google::protobuf::Message& proto) const = 0;
}; // class ProtobufSerializable

} // namespace lbann

#endif // LBANN_INCLUDE_LBANN_UTILS_PROTOBUF_SERIALIZABLE_HPP_INCLUDED
