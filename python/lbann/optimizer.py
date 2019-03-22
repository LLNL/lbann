from lbann import lbann_pb2
import lbann.util.class_generator

class Optimizer:
    def export_proto(self):
        """Construct and return a protobuf message."""
        return lbann_pb2.Optimizer()

# Generate Optimizer sub-classes from lbann.proto
# Note: The list of skip fields must be updated if any new fields are
# added to the Optimizer message in lbann.proto
classes = lbann.util.class_generator.generate_classes_from_protobuf_message(
    lbann_pb2.Optimizer,
    base_class = Optimizer,
    base_has_export_proto = True)
for c in classes:
    globals()[c.__name__] = c
