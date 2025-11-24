import numpy as np

from code_generator.operators import reduce
from code_generator.tflite import Model
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.ReducerOptions import ReducerOptions

from .utils import (
    get_input_tensors, 
    get_nhwc_from_shape,
    get_np_from_wrapper,
    get_output_tensors,
    getMultiplierShift,
    getTensorTypeStr
)

def parse_reduce(op, model: Model.Model, reduce_type: str):
    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) == 2, "input tensors length should be 2"
    input_tensor = input_tensors[0]
    axis = input_tensors[1]
    axis = get_np_from_wrapper(axis)

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

    # Reducer params
    assert op.BuiltinOptionsType() == BuiltinOptions.ReducerOptions
    op_options = op.BuiltinOptions()
    reducer_options = ReducerOptions()
    reducer_options.Init(op_options.Bytes, op_options.Pos)
    keep_dims = reducer_options.KeepDims()

    if (reduce_type not in ["max", "mean"]) or (keep_dims != False):
        raise NotImplementedError
    
    # Quantized mean params
    input_zero_point = input_tensor.qnn_params["zero_point"] \
        if input_tensor.qnn_params else 0
    input_scale = input_tensor.qnn_params["scale"] \
        if input_tensor.qnn_params else 0
    output_zero_point = output_tensor.qnn_params["zero_point"]
    output_scale = output_tensor.qnn_params["scale"]

    if output_dtype == "int8":
        min_val = -128
        max_val = 127
    else:
        raise NotImplementedError

    # Do as TFLM kernels/internal/reference/reduce.h:QuantizedMeanOrSum does
    effective_scale = np.double(input_scale) / np.double(output_scale)
    multiplier, shift = getMultiplierShift([effective_scale])
    assert len(multiplier) == 1 and len(shift) == 1
    multiplier = multiplier[0]
    shift = shift[0]
    if reduce_type == "mean":
        # Need to readapt the multiplier to 1/num_elements_in_axis
        num_elements_in_axis = 1
        input_size = [1, input_h, input_w, input_c]
        for ax in axis:
            num_elements_in_axis *= input_size[ax]
        clz_str = lambda s : len(s) - len(s.lstrip('0'))
        leading_zeroes = clz_str(np.binary_repr(num_elements_in_axis, width=64))
        extra_shift = 63 - leading_zeroes
        if extra_shift > 32: extra_shift = 32
        if extra_shift > 31 + shift: extra_shift = 31 + shift
        multiplier = int((np.int64(multiplier) << extra_shift) / num_elements_in_axis)
        shift -= extra_shift

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
        "reduce_type": reduce_type,
        "keep_dims": keep_dims,
        "axis": axis,
        "input_zero_point": input_zero_point,
        "output_scale": output_scale,
        "output_zero_point": output_zero_point,
        "multiplier": multiplier,
        "shift": shift,
        "max_val": max_val,
        "min_val": min_val,
    }

    op = reduce.Reduce(params)

    return op