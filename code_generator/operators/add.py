import warnings

from ..constant import USE_BIT_MASK
from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["Add"]

default_params = {
    # op related
    "op": "ADD",
    "input_idx": None,
    "input2_idx": None,
    "output_idx": None,
    # tensor related
    "input_dim": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "input2_dim": None,
    "input2_h": None,
    "input2_w": None,
    "input2_c": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "input_dtype": "int8",
    "input2_dtype": "int8",
    "output_dtype": "int8",
    # quantization related
    "input_zero_point": None,
    "input2_zero_point": None,
    "output_zero_point": None,
    "input_scale": None,
    "input2_scale": None,
    "output_scale": None,
    "input_multiplier": None,
    "input2_multiplier": None,
    "output_multiplier": None,
    "input_effective_scale": None,
    "input2_effective_scale": None,
    "output_effective_scale": None,
    "input_shift": None,
    "input2_shift": None,
    "output_shift": None,
    "left_shift": None,
    # fof Q training
    "need_Bmask": False,
    "output2_h": None,
    "output2_w": None,
    "output2_c": None,
    "output2_idx": None,
    "output2_dtype": "int8",
    # Shiming: output min and max
    "fused_activation_function": None,
    "output_activation_min": None,
    "output_activation_max": None,
    "broadcast": None,
    "broadcast_on_axis": None,
    "input2_const": None,
}


class Add(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()
        # handle input/output tensors in HWC format
        self._add_input(
            self.params["input_idx"],
            self.params["input_dtype"],
            self.params["input_c"],
            self.params["input_w"],
            self.params["input_h"],
        )
        self._add_input(
            self.params["input2_idx"],
            self.params["input2_dtype"],
            self.params["input2_c"],
            self.params["input2_w"],
            self.params["input2_h"],
        )
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

        if None in default_params:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def get_macs(self) -> int:
        p = self.params
        return p["output_h"] * p["output_w"] * p["output_c"]
    
    # Shiming: for ADD with constant input
    def get_weights_size(self) -> int:
        p = self.params
        if p["input2_const"] is None: return 0
        if p["input_dtype"] in {"float32", "fp32"}:
            size = 4
        else:
            size = 1
        return p["input2_h"] * p["input2_w"] * p["input2_c"] * size

    # Shiming: Support non-fp-requantized version
    def generate_inference_str(
        self,
        fp_requantize: bool = False,
    ):
        string = ""
        params = self.params
        if fp_requantize:
            if params["need_Bmask"]:
                if USE_BIT_MASK:
                    string += (
                        f"add_fpreq_bitmask({str(int(params['input_h']*params['input_w']*params['input_c']))}, "
                        + f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                    )
                else:
                    string += f"add_fpreq_mask({str(int(params['input_h']*params['input_w']*params['input_c']))}, "
                    +f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                string += (
                    f"{str(params['input_scale'])},{str(params['input_zero_point'])},"
                    + f"{self._getBufferstr(params['input2_buf_add'], params['input2_buf_add_offset'])},"
                    + f"{str(params['input2_scale'])},{str(params['input2_zero_point'])},"
                    + f"{str(params['output_scale'])},{str(params['output_zero_point'])},"
                    + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])},"
                    + f"{self._getBufferstr(params['output2_buf_add'], params['output2_buf_add_offset'])});\n"
                )
            else:
                string += (
                    f"add_fpreq({str(int(params['input_h']*params['input_w']*params['input_c']))}, "
                    + f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                    + f"{str(params['input_scale'])},{str(params['input_zero_point'])},"
                    + f"{self._getBufferstr(params['input2_buf_add'], params['input2_buf_add_offset'])},"
                    + f"{str(params['input2_scale'])},{str(params['input2_zero_point'])},"
                    + f"{str(params['output_scale'])},{str(params['output_zero_point'])},"
                    + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
                )
        else:
            # Construct ADD_params right here
            string += (
                f"add_params.left_shift = {params['left_shift']};\n"
                f"add_params.input1_offset = {params['input_zero_point'] * -1};\n"
                f"add_params.input1_multiplier = {params['input_multiplier']};\n"
                f"add_params.input1_shift = {params['input_shift']};\n"
                f"add_params.input2_offset = {params['input2_zero_point'] * -1};\n"
                f"add_params.input2_multiplier = {params['input2_multiplier']};\n"
                f"add_params.input2_shift = {params['input2_shift']};\n"
                f"add_params.output_offset = {params['output_zero_point']};\n"
                f"add_params.output_multiplier = {params['output_multiplier']};\n"
                f"add_params.output_shift = {params['output_shift']};\n"
                f"add_params.quantized_activation_min = {params['output_activation_min']};\n"
                f"add_params.quantized_activation_max = {params['output_activation_max']};\n"
            )
            func_name = "add"
            size_info = str(int(params['input_h']*params['input_w']*params['input_c']))
            if params["broadcast"] and (params["broadcast_on_axis"] == [1, 2]):
                func_name += "_broadcast_axis_1_2"
                size_info = "{:d}, {:d}, {:d}".format(params['input_h'], params['input_w'], params['input_c'])
            string += (
                f"{func_name}({size_info}, &add_params, "
                + f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
            )
            if params["input2_const"] is not None:
                string += f"constinput_{params['parsed_trainable']},"
            else:
                string += f"{self._getBufferstr(params['input2_buf_add'], params['input2_buf_add_offset'])},"
            string += f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
            
        return string
