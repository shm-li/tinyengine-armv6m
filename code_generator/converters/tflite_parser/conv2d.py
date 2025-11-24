import math

import numpy as np

from code_generator.operators import conv2d, depthwiseConv2d
from code_generator.tflite import Model
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.Conv2DOptions import Conv2DOptions
from code_generator.tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions

from decimal import Decimal, ROUND_HALF_UP

from .utils import (
    # Shiming:
    compute_out_size,
    compute_padding,
    get_input_tensors,
    get_np_from_wrapper,
    get_output_tensors,
    getMultiplierShift,
    getOpCodeStr,
    getTensorTypeStr,
)


def parse_conv2d(op, model: Model.Model, tmpPADIndice=None):
    # operator
    op_code_str = getOpCodeStr(op, model)

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, model)
    input_tensor_count = len(input_tensors)
    assert input_tensor_count >= 2, "input tensors length should be >= 2"

    input_tensor = input_tensors[0]
    weight_tensor = input_tensors[1]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # conv_2d options
    if op_code_str == "CONV_2D":
        assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
        op_options = op.BuiltinOptions()
        conv_options = Conv2DOptions()
        conv_options.Init(op_options.Bytes, op_options.Pos)
    if op_code_str == "DEPTHWISE_CONV_2D":
        assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
        op_options = op.BuiltinOptions()
        conv_options = DepthwiseConv2DOptions()
        conv_options.Init(op_options.Bytes, op_options.Pos)

    # conv parameters
    stride_h = conv_options.StrideH()
    stride_w = conv_options.StrideW()

    # shapes
    _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
    if op_code_str == "CONV_2D":
        output_c, kernel_h, kernel_w, _ = weight_tensor.tensor.ShapeAsNumpy()
    elif op_code_str == "DEPTHWISE_CONV_2D":
        _, kernel_h, kernel_w, output_c = weight_tensor.tensor.ShapeAsNumpy()
    _, output_h, output_w, output_c_dual = output_tensor.tensor.ShapeAsNumpy()
    assert output_c_dual == output_c, "output channels not match"

    # Shiming: read dilation factors
    dilation_h = conv_options.DilationHFactor()
    dilation_w = conv_options.DilationWFactor()
    if dilation_w != 1 or dilation_h != 1:
        raise NotImplementedError("No support for dilation factors > 1")

    # tensor types
    input_type = getTensorTypeStr(input_tensor.tensor.Type())
    output_type = getTensorTypeStr(output_tensor.tensor.Type())
    weight_type = getTensorTypeStr(weight_tensor.tensor.Type())
    assert input_type == output_type == weight_type, "tensor type not consistent"

    # tensor value: weight, scalers
    weight_value = get_np_from_wrapper(weight_tensor)
    if input_tensor_count == 3:
        bias_tensor = input_tensors[2]
        # bias = self._get_np_from_wrapper(bias_tensor).astype('int') # forcely casting for testing latency
        bias = get_np_from_wrapper(bias_tensor)
    else:
        bias = None

    # quantized setting
    input_zero_point = input_tensor.qnn_params["zero_point"]
    output_zero_point = output_tensor.qnn_params["zero_point"]
    input_scale = input_tensor.qnn_params["scale"]
    weight_scale = weight_tensor.qnn_params["scale"]
    output_scale = output_tensor.qnn_params["scale"]
    effective_scale = np.double(input_scale) * np.double(weight_scale) / np.double(output_scale)

    # quantized inference, used for requantize
    multiplier, shift = getMultiplierShift(effective_scale)

    # find previous layer and redirct the index and fuse pad into conv
    # Shiming: fixing the calculation
    if tmpPADIndice is not None:
        if tmpPADIndice.output_idx == input_tensor.tensor_idx:
            input_idx = tmpPADIndice.input_idx
            #input_h = input_h - math.floor(kernel_h / 2) * 2
            #input_w = input_w - math.floor(kernel_h / 2) * 2
            _, last_input_h, last_input_w, last_input_c = \
                    tmpPADIndice.last_input_tensor.tensor.ShapeAsNumpy()
            paddings = get_np_from_wrapper(tmpPADIndice.paddings)
            last_layer_padding_h = paddings[1][0]
            last_layer_padding_h_offset = paddings[1][1] - paddings[1][0]
            last_layer_padding_w = paddings[2][0]
            last_layer_padding_w_offset = paddings[2][1] - paddings[2][0]
            print("Fusing last PAD layer with conv. changing input size from", 
                  input_h, input_w, input_c, 
                  "to", 
                  last_input_h, last_input_w, input_c)
            input_h = last_input_h
            input_w = last_input_w
            if input_c != last_input_c:
                raise NotImplementedError
        else:
            input_idx = input_tensor.tensor_idx
    else:
        input_idx = input_tensor.tensor_idx

    # Shiming: after fixing input size with fused padding layer, 
    #   calculate padding. 
    # TinyEngine's original computation for padding is incorrect; using my own
    padding = conv_options.Padding() # padding type. 0: SAME, 1: VALID (no pad)
    if (tmpPADIndice is not None) and (tmpPADIndice.output_idx == input_tensor.tensor_idx):
        # If last layer is "PAD", force-set layer padding type to "SAME"
        padding = 0 
    padding_h = 0
    padding_w = 0

    test_out_h = compute_out_size(padding, input_h, kernel_h, stride_h)
    test_out_w = compute_out_size(padding, input_w, kernel_w, stride_w)
    assert (test_out_h == output_h) and (test_out_w == output_w)
    padding_h, padding_h_offset = compute_padding(stride_h, dilation_h, input_h, 
                                kernel_h, output_h)
    padding_w, padding_w_offset = compute_padding(stride_w, dilation_w, input_w, 
                                kernel_w, output_w)
    if (tmpPADIndice is not None) and (tmpPADIndice.output_idx == input_tensor.tensor_idx):
        # If last layer is "PAD", check if our calculated padding matches it
        # assert padding_h == last_layer_padding_h
        # assert padding_h_offset == last_layer_padding_h_offset
        # assert padding_w == last_layer_padding_w
        # assert padding_w_offset == last_layer_padding_w_offset
        # Actually, it might happen that there are redundant padding in PAD layer
        print("Using last layer's padding:", 
              last_layer_padding_h, last_layer_padding_w,
              last_layer_padding_h_offset, last_layer_padding_w_offset
        )
        padding_h = last_layer_padding_h
        padding_w = last_layer_padding_w
        padding_h_offset = last_layer_padding_h_offset
        padding_w_offset = last_layer_padding_w_offset
    # print("padding:", padding, padding_h, padding_w)
    if padding_h != padding_w or padding_h_offset != padding_w_offset:
        raise NotImplementedError("Please implement your own conv function"
                                  "for unequal padding h and w "
                                  "%d, %d".format(padding_h, padding_w))
    
    # Shiming: Activation
    # 0: NONE, 1: RELU, 3: RELU6
    # Normally the output_activation_min/max is quantized to -128/127.
    #   We just do a sanity check here. Params are not actually used. Yet TODO.
    fused_act_func = conv_options.FusedActivationFunction()
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

    params = {
        # operator
        "op": op_code_str,
        # conv
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        #   Shiming: fixing padding
        "padding": "VALID" if padding else "SAME", #math.floor(kernel_h / 2),
        "padding_h": padding_h,
        "padding_w": padding_w,
        "padding_h_offset": padding_h_offset,
        "padding_w_offset": padding_w_offset,
        "stride_h": stride_h,
        "stride_w": stride_w,
        # tensor
        "input_idx": input_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_dim": 3,
        "output_dim": 3,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "dtypte": input_type,
        # trainable parameters
        "weight_value": weight_value,
        "bias": bias,
        "effective_scale": effective_scale,
        "input_zero_point": input_zero_point,
        "output_zero_point": output_zero_point,
        "input_scale": input_scale,
        "weight_scale": weight_scale,
        "output_scale": output_scale,
        # quantized infernece
        "multiplier": multiplier,
        "shift": shift,
        # Shiming: add activation function params
        "fused_activation_function": fused_act_func,
        "output_activation_min": output_activation_min,
        "output_activation_max": output_activation_max,
    }

    if op_code_str == "CONV_2D":
        op = conv2d.Conv2d(params)
    elif op_code_str == "DEPTHWISE_CONV_2D":
        op = depthwiseConv2d.DepthwiseConv2d(params)

    return op
