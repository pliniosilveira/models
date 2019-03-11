import onnx
import warnings
from onnx_tf.backend import prepare, run_model

import numpy as np
import os
import glob

from onnx import numpy_helper

warnings.filterwarnings('ignore') # Ignore all the warning messages in this tutorial

MODEL_PATH = 'resnet50/model.onnx'

print ("Loading model")

model = onnx.load(MODEL_PATH) # Load the ONNX file
tf_rep = prepare(model) # Import the ONNX model to Tensorflow

print ("Model loaded")

#model = onnx.load('model.onnx')
test_data_dir = 'resnet50/test_data_set_0'

# Load inputs
print("Loading inputs")
inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))
print("Inputs loaded")

# Load reference outputs
print("Loading outputs")
ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
for i in range(ref_outputs_num):
    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))
print("Outputs loaded")

# Run the model on the backend
print("Running model for {} inputs".format(len(inputs))) 
outputs = list(run_model(model, inputs))
print("Model executed")

# Compare the results with reference outputs.
print("Comparing outputs")
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o)
print("Outputs match")

print("FINISHED")
