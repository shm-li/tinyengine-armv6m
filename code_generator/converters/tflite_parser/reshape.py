
from code_generator.operators import reshape
from code_generator.tflite import Model

from .utils import (
    get_input_tensors, 
    get_nhwc_from_shape,
    get_np_from_wrapper,
    get_output_tensors,
    getTensorTypeStr
)

def parse_reshape(op, model: Model.Model):
    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) == 2, "input tensors length should be 1"
    input_tensor = input_tensors[0]
    # Second tensor is reshape shape
    shape_tensor = input_tensors[1]
    # Don't know if shape has value or not yet...
    # shape = get_np_from_wrapper(shape_tensor)

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]
    
    # data types
    input_dtype = getTensorTypeStr(input_tensor.tensor.Type())
    output_dtype = getTensorTypeStr(output_tensor.tensor.Type())

    assert input_dtype == output_dtype

    # shapes
    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())

    # We only support the case where the total length doesn't change
    # In this case, we connect the last layers' output to this layer's output.
    # Do nothing in code generation.
    keep_length = (input_h * input_w * input_c 
                   == output_h * output_w * output_c)
    if not keep_length:
        raise NotImplementedError

    params = {
        "input_idx": input_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "input_dtype": input_dtype,
        "output_idx": output_tensor.tensor_idx,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "output_dtype": output_dtype,
        # Shiming: fill in shape info
        "shape_idx": shape_tensor.tensor_idx,
        # "shape": shape,
        # Shiming: if the layer would change number of elems
        "keep_length": keep_length,
    }

    op = reshape.Reshape(params)

    return op