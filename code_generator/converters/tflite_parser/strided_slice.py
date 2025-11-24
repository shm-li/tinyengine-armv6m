from code_generator.operators import strided_slice
from code_generator.tflite import Model

from .utils import (
    get_input_tensors, 
    get_nhwc_from_shape,
    get_np_from_wrapper,
    get_output_tensors,
    getTensorTypeStr
)

def parse_strided_slice(op, model: Model.Model):
    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) == 4, "input tensors length should be 4"
    input_tensor = input_tensors[0]
    begin = get_np_from_wrapper(input_tensors[1])[0]
    end = get_np_from_wrapper(input_tensors[2])[0]
    strides = get_np_from_wrapper(input_tensors[3])[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]
    
    # data types
    input_dtype = getTensorTypeStr(input_tensor.tensor.Type())
    output_dtype = getTensorTypeStr(output_tensor.tensor.Type())

    assert input_dtype == output_dtype

    # shapes
    input_n, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    output_n, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())

    params = {
        "input_idx": input_tensor.tensor_idx,
        "d1": input_h,
        "d2": input_h,
        "d3": input_w,
        "d4": input_c,
        "input_dtype": input_dtype,
        "output_idx": output_tensor.tensor_idx,
        "o_d1": output_n,
        "o_d2": output_h,
        "o_d3": output_w,
        "o_d4": output_c,
        "output_dtype": output_dtype,
        "begin": begin,
        "end": end,
        "strides": strides,
    }

    op = strided_slice.StridedSlice(params)

    return op