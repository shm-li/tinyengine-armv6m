import math
import numpy as np

from code_generator.operators import quantize
from code_generator.tflite import Model
from decimal import Decimal, ROUND_HALF_UP

from .utils import (
    get_input_tensors, 
    get_nhwc_from_shape,
    get_output_tensors,
    getMultiplierShift,
    getTensorTypeStr
)


def parse_quantize(op, model: Model.Model):
    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) == 1, "input tensors length should be 1"
    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]
    
    # data types
    input_dtype = getTensorTypeStr(input_tensor.tensor.Type())
    output_dtype = getTensorTypeStr(output_tensor.tensor.Type())

    # shapes
    _, input_h, input_w, input_c = get_nhwc_from_shape(input_tensor.tensor.ShapeAsNumpy())
    _, output_h, output_w, output_c = get_nhwc_from_shape(output_tensor.tensor.ShapeAsNumpy())

    # quantization params
    input_zero_point = input_tensor.qnn_params["zero_point"] \
        if input_tensor.qnn_params else 0
    input_scale = input_tensor.qnn_params["scale"] \
        if input_tensor.qnn_params else 0

    output_zero_point = output_tensor.qnn_params["zero_point"]
    output_scale = output_tensor.qnn_params["scale"]

    # quantization params that needs further parsing
    if output_dtype == "int8":
        min_val = -128
        max_val = 127
    elif output_dtype == "uint8":
        min_val = 0
        max_val = 255

    if input_dtype == "float32":
        requantize = False
        # mantissa_fixed = 0
        # exponent = 0
        multiplier = 0
        shift = 0
    else:
        requantize = True
        # Do the same thing as TFLM
        # effective_scale = input_scale / output_scale
        # mantissa, exponent = math.frexp(effective_scale)
        # assert mantissa < 1
        # # mantissa_fixed = round(mantissa * (1 << 31))
        # mantissa_fixed = int(Decimal(mantissa * (1 << 31)).quantize(Decimal(1), 
        #                                             rounding=ROUND_HALF_UP))
        # if mantissa_fixed == (1 << 31):
        #     mantissa_fixed /= 2
        #     exponent += 1
        # assert mantissa_fixed <= 2147483647 # int32 max
        # if exponent < -31:
        #     exponent = 0
        #     mantissa_fixed = 0
        # Actually TinyEngine has its own getMultiplierShift
        effective_scale = np.double(input_scale) / np.double(output_scale)
        multiplier, shift = getMultiplierShift([effective_scale])
        assert len(multiplier) == 1 and len(shift) == 1
        multiplier = multiplier[0]
        shift = shift[0]
    
    params = {
        "input_idx": input_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "input_zero_point": input_zero_point,
        "input_scale": input_scale,
        "input_dtype": input_dtype,
        "output_idx": output_tensor.tensor_idx,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "output_zero_point": output_zero_point,
        "output_scale": output_scale,
        "output_dtype": output_dtype,
        "requantize": requantize,
        "min_val": min_val,
        "max_val": max_val,
        "multiplier": multiplier,
        "shift": shift,
    }

    # print(output_zero_point, output_scale, input_dtype, output_dtype, 
    #       exponent, mantissa_fixed)

    op = quantize.Quantize(params)

    return op
