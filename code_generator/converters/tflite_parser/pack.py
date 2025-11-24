from code_generator.operators import pack
from code_generator.tflite import Model

from .utils import (
    get_input_tensors, 
    get_nhwc_from_shape,
    get_np_from_wrapper,
    get_output_tensors,
    getTensorTypeStr
)

def parse_pack(op, model: Model.Model):
    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) > 1, "input tensors length should be >1"

    input_tensor = input_tensors[0]
    pack_tensor = []
    for each in input_tensors[1:]:
        pack_tensor.append(get_np_from_wrapper(each))

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
        "pack_tensor": pack_tensor,
    }

    op = pack.Pack(params)

    return op