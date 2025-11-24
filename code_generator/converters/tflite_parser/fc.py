import numpy as np

from code_generator.operators import conv2d
from code_generator.tflite import Model
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.FullyConnectedOptions import FullyConnectedOptions

from decimal import Decimal, ROUND_HALF_UP

from .utils import (
    get_input_tensors,
    get_nhwc_from_shape,
    get_np_from_wrapper,
    get_output_tensors,
    getMultiplierShift,
    getTensorTypeStr,
)


def parse_fc(op, model: Model.Model):
    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, model)
    input_tensor_count = len(input_tensors)
    assert input_tensor_count == 3, "input tensors length should be 3"

    input_tensor = input_tensors[0]
    weight_tensor = input_tensors[1]
    bias_tensor = input_tensors[2]
    weight = get_np_from_wrapper(weight_tensor)
    # Shiming: bias can be None
    if bias_tensor.tensor:
        bias = get_np_from_wrapper(bias_tensor)
    else:
        # bias is None, but we construct a zero vector
        bias = np.zeros(weight_tensor.tensor.ShapeAsNumpy()[0], dtype=np.int32)

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # shapes
    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    _, _, output_c, input_c_dual = get_nhwc_from_shape(weight_tensor.tensor.ShapeAsNumpy())
    _, _, output_h, output_c_dual = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())
    assert input_c_dual == input_c, "channels not match"
    assert output_c_dual == output_c, "channels not match"

    # tensor types
    input_type = getTensorTypeStr(input_tensor.tensor.Type())
    output_type = getTensorTypeStr(output_tensor.tensor.Type())
    assert input_type == output_type, "tensor type not consistent"

    # initialize quantized parameters as None for floating-pointer ops
    input_zero_point = None
    input_scale = None
    weight_scale = None
    bias_scale = None
    output_zero_point = None
    output_scale = None
    multiplier = None
    shift = None
    effective_scale = None

    # quantized setting
    if "float32" not in [input_type, output_type]:
        input_zero_point = input_tensor.qnn_params["zero_point"]
        output_zero_point = output_tensor.qnn_params["zero_point"]
        input_scale = input_tensor.qnn_params["scale"]
        weight_scale = weight_tensor.qnn_params["scale"]
        bias_scale = bias_tensor.qnn_params["scale"]
        output_scale = output_tensor.qnn_params["scale"]

        # We support per channel in the CONV2D operator
        if isinstance(bias_scale, float) and isinstance(weight_scale, float):
            np_ones = np.ones(output_c)
            bias_scale = np_ones * bias_scale
            np_ones = np.ones(output_c)
            output_scale = np_ones * output_scale
        effective_scale = np.double(input_scale) * np.double(weight_scale) / np.double(output_scale)

        # follows tensorflow lite micro
        multiplier, shift = getMultiplierShift(effective_scale)
    
    # Shiming: Activation
    # 0: NONE, 1: RELU, 3: RELU6
    # Normally the output_activation_min/max is quantized to -128/127.
    #   We do a sanity check here. We yet have no impl for calculating the
    #   correct output_acitvation_min/max if the check fails, but it is easy
    op_options = op.BuiltinOptions()
    fc_options = FullyConnectedOptions()
    fc_options.Init(op_options.Bytes, op_options.Pos)
    fused_act_func = fc_options.FusedActivationFunction()
    one_output_scale = output_scale[0]
    if fused_act_func == 1 or fused_act_func == 3:
        quantized_0 = Decimal(0 / one_output_scale).\
            quantize(Decimal(1), rounding=ROUND_HALF_UP) + output_zero_point
        if int(quantized_0) > -128:
            print("WARNING: ACT_MIN is {}".format(quantized_0))
    if fused_act_func == 3:
        quantized_6 = Decimal(6 / one_output_scale).\
            quantize(Decimal(1), rounding=ROUND_HALF_UP) + output_zero_point
        if int(quantized_6) < 127:
            print("WARNING: ACT_MAX is {}".format(quantized_6))

    params = {
        # operator
        "op": "CONV_2D",
        # tensor
        "input_idx": input_tensor.tensor_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "input_dim": 3,
        "output_dim": 2,
        "output_h": output_h,
        "output_w": 1,
        "output_c": output_c,
        "dtypte": input_type,
        "kernel_h": 1,
        "kernel_w": 1,
        #   Shiming: adding padding and stride params
        "padding_h": 0,
        "padding_w": 0,
        "padding_h_offset": 0,
        "padding_w_offset": 0,
        "stride_h": 1,
        "stride_w": 1,
        # trainable parameters
        "weight_value": weight,
        "bias": bias,
        "effective_scale": effective_scale,
        "input_zero_point": input_zero_point,
        "output_zero_point": output_zero_point,
        "input_scale": input_scale,
        "output_scale": output_scale,
        # quantized infernece
        "multiplier": multiplier,
        "shift": shift,
        # Shiming: add activation function params
        "fused_activation_function": fused_act_func,
    }

    op = conv2d.Conv2d(params)

    return op
