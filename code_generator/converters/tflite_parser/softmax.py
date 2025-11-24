import numpy as np

from code_generator.operators import softmax
from code_generator.tflite import Model
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.SoftmaxOptions import SoftmaxOptions

from .utils import (
    get_input_tensors, 
    get_nhwc_from_shape,
    get_output_tensors,
    getMultiplierShift,
    getTensorTypeStr
)

def parse_softmax(op, model: Model.Model):
    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) == 1, "input tensors length should be 1"
    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]
    
    # data types
    input_dtype = getTensorTypeStr(input_tensor.tensor.Type())
    output_dtype = getTensorTypeStr(output_tensor.tensor.Type())

    if input_dtype == "int8" and output_dtype == "int8":
        pass
        # effective_scale = np.double(input_scale) / np.double(output_scale)
        # multiplier, shift = getMultiplierShift([effective_scale])
    else:
        raise NotImplementedError("No impl for non-int8 Softmax")

    # shapes
    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())

    if input_h * input_w * input_c != output_h * output_w * output_c:
        raise NotImplementedError("No impl for a softmax layer with "
                                  "different input and output shape")

    # quantization params
    input_zero_point = input_tensor.qnn_params["zero_point"] \
        if input_tensor.qnn_params else 0
    input_scale = input_tensor.qnn_params["scale"] \
        if input_tensor.qnn_params else 0

    output_zero_point = output_tensor.qnn_params["zero_point"]
    output_scale = output_tensor.qnn_params["scale"]

    
    # Softmax params
    assert op.BuiltinOptionsType() == BuiltinOptions.SoftmaxOptions
    op_options = op.BuiltinOptions()
    softmax_options = SoftmaxOptions()
    softmax_options.Init(op_options.Bytes, op_options.Pos)
    beta = np.double(softmax_options.Beta())
    # Do the same thing as TFLM softmax_common.cc
    # Calculate input_multiplier and input_left_shift:
    input_integer_bits = 5
    input_beta_real_multiplier = beta * input_scale * (1 << (31 - input_integer_bits))
    max_real_multiplier = np.double((1 << 31) - 1)
    if input_beta_real_multiplier > max_real_multiplier: input_beta_real_multiplier = max_real_multiplier
    assert input_beta_real_multiplier > 1
    input_multiplier, input_left_shift = getMultiplierShift([input_beta_real_multiplier])
    assert len(input_multiplier) == 1 and len(input_left_shift) == 1
    input_multiplier = input_multiplier[0]
    input_left_shift = input_left_shift[0]
    # Calculate diff_min:
    diff_min = ((1 << input_integer_bits) - 1) * np.double((1 << (31 - input_integer_bits))) / np.double((1 << input_left_shift))
    diff_min = -1 * np.floor(diff_min).astype(int)
    # print("Softmax options:", dir(softmax_options), beta, input_multiplier, input_left_shift, diff_min)

    params = {
        "input_idx": input_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "input_dtype": input_dtype,
        "input_zero_point": input_zero_point,
        "input_scale": input_scale,
        "output_idx": output_tensor.tensor_idx,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "output_dtype": output_dtype,
        "output_zero_point": output_zero_point,
        "output_scale": output_scale,
        "beta": beta,
        "input_multiplier": input_multiplier,
        "input_left_shift": input_left_shift,
        "diff_min": diff_min,
    }

    op = softmax.Softmax(params)

    return op