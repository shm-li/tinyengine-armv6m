
from code_generator.operators import transpose
from code_generator.tflite import Model

from .utils import (
    get_input_tensors, 
    get_nhwc_from_shape,
    get_np_from_wrapper,
    get_output_tensors,
    getTensorTypeStr
)

def parse_transpose(op, model: Model.Model):
    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) == 2, "input tensors length should be 2"
    input_tensor = input_tensors[0]
    # Second tensor is permutation
    perm = input_tensors[1]
    perm = get_np_from_wrapper(perm)

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
        "perm": perm,
    }

    op = transpose.Transpose(params)

    return op