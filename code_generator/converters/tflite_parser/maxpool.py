from code_generator.operators import maxpool2d
from code_generator.tflite import Model
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.Pool2DOptions import Pool2DOptions

from .utils import (
    # Shiming:
    compute_padding,
    get_input_tensors,
    get_np_from_wrapper,
    get_output_tensors
)


def parse_maxpool(op, model: Model.Model, tmpPADIndice=None):
    # Incase no params
    input_type = None
    input_zero_point = None
    output_zero_point = None
    input_scale = None
    output_scale = None

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, model)
    input_tensor_count = len(input_tensors)
    assert input_tensor_count == 1, "input tensors length should be 1"

    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # Shiming: quantized setting
    input_zero_point = input_tensor.qnn_params["zero_point"]
    output_zero_point = output_tensor.qnn_params["zero_point"]
    input_scale = input_tensor.qnn_params["scale"]
    output_scale = output_tensor.qnn_params["scale"]

    # shapes
    _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
    _, output_h, output_w, output_c = output_tensor.tensor.ShapeAsNumpy()

    # pool parameters
    assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
    op_options = op.BuiltinOptions()
    pool2d_options = Pool2DOptions()
    pool2d_options.Init(op_options.Bytes, op_options.Pos)
    stride_h = pool2d_options.StrideH()
    stride_w = pool2d_options.StrideW()
    filter_h = pool2d_options.FilterHeight()
    filter_w = pool2d_options.FilterWidth()
    
    # Shimng: find previous layer and redirct the index and fuse pad into conv
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
            print("Fusing last PAD layer with max_pool. changing input size "
                  "from", 
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
    
    # Shiming:
    padding_h, padding_h_offset = compute_padding(stride_h, 1, input_h, 
                                filter_h, output_h)
    padding_w, padding_w_offset = compute_padding(stride_w, 1, input_w, 
                                filter_w, output_w)
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

    pool_params = {
        # operator
        "op": "MAX_POOL_2D",
        # pool parameters
        "filter_h": filter_h,
        "filter_w": filter_w,
        "stride_h": stride_h,
        "stride_w": stride_w,
        # Shiming:
        "padding_h": padding_h,
        "padding_w": padding_w,
        "padding_h_offset": padding_h_offset,
        "padding_w_offset": padding_w_offset,
        # tensor
        "input_idx": input_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_h": input_h,
        "input_w": input_w,
        "input_c": input_c,
        "input_dim": 3,
        "output_dim": 3,
        "output_h": output_h,
        "output_w": output_w,
        "output_c": output_c,
        "dtype": input_type,
        # trainable parameters
        "input_zero_point": input_zero_point,
        "output_zero_point": output_zero_point,
        "input_scale": input_scale,
        "output_scale": output_scale,
    }
    op = maxpool2d.maxPool2d(pool_params)

    return op
