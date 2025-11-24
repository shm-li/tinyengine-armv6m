import math

import numpy as np

from code_generator.operators import mul
from code_generator.tflite import Model
from code_generator.tflite.MulOptions import MulOptions

from decimal import Decimal, ROUND_HALF_UP

from .utils import (
    get_input_tensors, 
    get_nhwc_from_shape, 
    get_np_from_wrapper,
    get_output_tensors, 
    getMultiplierShift,
    getOpCodeStr, 
    getTensorTypeStr
)


def parse_mul(op, model: Model.Model):
    # operator
    op_code_str = getOpCodeStr(op, model)

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, model)
    input_tensor_count = len(input_tensors)
    assert input_tensor_count == 2, "input should be 2 tensors"

    input_tensor = input_tensors[0]
    input2_tensor = input_tensors[1]
    input2_const = get_np_from_wrapper(input2_tensor)

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # shapes
    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    _, input2_h, input2_w, input2_c = get_nhwc_from_shape(input2_tensor.tensor.ShapeAsNumpy())
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())

    broadcast = False
    broadcast_on_axis = [1, 2, 3]
    if (
        input_h == input2_h == output_h
        and input_w == input2_w == output_w
        and input_c == input2_c == output_c
    ):
        # common mul
        raise NotImplementedError
    else:
        broadcast = True
        if input_h == input2_h == output_h:
            broadcast_on_axis.remove(1)
        if input_w == input2_w == output_w:
            broadcast_on_axis.remove(2)
        if input_c == input2_c == output_c:
            broadcast_on_axis.remove(3)
        assert broadcast_on_axis != []
        # broadcast mul
    #assert input_h == input2_h == output_h, "tensor shpae not consistent"
    #assert input_w == input2_w == output_w, "tensor shpae not consistent"
    #assert input_c == input2_c == output_c, "tensor shpae not consistent"

    # tensor types
    input_type = getTensorTypeStr(input_tensor.tensor.Type())
    input_type2 = getTensorTypeStr(input2_tensor.tensor.Type())
    output_type = getTensorTypeStr(output_tensor.tensor.Type())
    assert input_type == input_type2 == output_type, "tensor type not consistent"

    # initialize quantized parameters as None for floating-pointer ops
    input_zero_point = None
    input_scale = None
    input2_zero_point = None
    input2_scale = None
    output_zero_point = None
    output_scale = None

    left_shift = None
    input_multiplier = None
    input_shift = None
    input2_multiplier = None
    input2_shift = None
    output_multiplier = None
    output_shift = None
    
    # quantized setting
    if input_type != "float32":
        input_zero_point = input_tensor.qnn_params["zero_point"]
        input_scale = input_tensor.qnn_params["scale"]
    if input_type2 != "float32":
        input2_zero_point = input2_tensor.qnn_params["zero_point"]
        input2_scale = input2_tensor.qnn_params["scale"]
    if output_type != "float32":
        output_zero_point = output_tensor.qnn_params["zero_point"]
        output_scale = output_tensor.qnn_params["scale"]

    if "float32" not in [output_type, input_type, input_type2]:
        # get multipliers and shifts
        real_output_scale = np.double(input_scale * input2_scale / output_scale)

        input_multiplier, input_shift = getMultiplierShift([input_scale])
        input_multiplier = input_multiplier[0]
        input_shift = input_shift[0]
        input2_multiplier, input2_shift = getMultiplierShift([input2_scale])
        input2_multiplier = input2_multiplier[0]
        input2_shift = input2_shift[0]
        output_multiplier, output_shift = getMultiplierShift([real_output_scale])
        output_multiplier = output_multiplier[0]
        output_shift = output_shift[0]
    
    # Shiming: honestly check output min and max!!
    op_options = op.BuiltinOptions()
    mul_options = MulOptions()
    mul_options.Init(op_options.Bytes, op_options.Pos)
    fused_act_func = mul_options.FusedActivationFunction()
    if output_type == "int8":
        output_activation_min = -128
        output_activation_max = 127
        if fused_act_func == 1 or fused_act_func == 3:
            quantized_0 = Decimal(0 / output_scale).\
                quantize(Decimal(1), rounding=ROUND_HALF_UP) + output_zero_point
            if int(quantized_0) > output_activation_min:
                raise RuntimeError("WARNING: ACT_MIN is {}".format(quantized_0))
                output_activation_min = int(quantized_0)
        if fused_act_func == 3:
            quantized_6 = Decimal(6 / output_scale).\
                quantize(Decimal(1), rounding=ROUND_HALF_UP) + output_zero_point
            if int(quantized_6) < output_activation_max:
                raise RuntimeError("WARNING: ACT_MAX is {}".format(quantized_6))
                output_activation_max = int(quantized_6)
    else:
        output_activation_min = None
        output_activation_max = None

    # assign params
    params = {
        # operator
        "op": op_code_str,
        # tensor
        "input_dtype": input_type,
        "input2_dtype": input_type2,
        "output_dtype": output_type,
        "input_idx": input_tensor.tensor_idx,
        "input2_idx": input2_tensor.tensor_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "input2_h": input_h,
        "input2_w": input_w,
        "input2_c": input_c,
        "input_dim": 3,
        "input2_dim": 3,
        "output_dim": 3,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "dtypte": input_type,
        # trainable parameters
        "input_zero_point": input_zero_point,
        "input2_zero_point": input2_zero_point,
        "output_zero_point": output_zero_point,
        "input_scale": input_scale,
        "input2_scale": input2_scale,
        "output_scale": output_scale,
        # quantized infernece
        "left_shift": left_shift,
        "input_multiplier": input_multiplier,
        "input2_multiplier": input2_multiplier,
        "input_shift": input_shift,
        "input2_shift": input2_shift,
        "output_multiplier": output_multiplier,
        "output_shift": output_shift,
        # Shiming: output min and max
        "fused_activation_function": fused_act_func,
        "output_activation_min": output_activation_min,
        "output_activation_max": output_activation_max,
        # Shiming: broadcast
        "broadcast": broadcast,
        "broadcast_on_axis": broadcast_on_axis,
        # Shiming: might be mul with const
        "input2_const": input2_const,
    }
    op = mul.Mul(params)

    return op

