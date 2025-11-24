import warnings

from .basic_utils import basicOperator, deep_copy_dicts, islabelstr, isParamstr, overwrite_dicts

__all__ = ["mul"]

default_params = {
    # op related
    "op": "MUL",
    "input_idx": None,
    "input2_idx": None,
    "output_idx": None,
    # tensor related
    "input_size": None,
    "input2_size": None,
    "output_size": None,
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
    "input_dtype": None,
    "input2_dtype": None,
    "output_dtype": None,
    # quantization related
    "weight_value": None,
    "bias": None,
    "input_zero_point": None,
    "input2_zero_point": None,
    "output_zero_point": None,
    "input_scale": None,
    "input2_scale": None,
    "output_scale": None,
    "input_multiplier": None,
    "input2_multiplier": None,
    "input_shift": None,
    "input2_shift": None,
    "output_multiplier": None,
    "output_shift": None,
    # input of scale from some conv2d
    "scale_conv_2d_op": None,
    "scale_from_add": None,
    "constant": None,
    # inplace
    "inplace": False,
    # Shiming:
    "fused_activation_function": None,
    "output_activation_min": None,
    "output_activation_max": None,
    "broadcast": None,
    "broadcast_on_axis": None,
    "input2_const": None,
}


class Mul(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()
        # handle input/output tensors in HWC format
        # self._add_input(self.params["input_idx"], self.params["input_dtype"], self.params["input_size"], 1, 1)
        # if not (isParamstr(self.params["input2_idx"]) or islabelstr(self.params["input2_idx"])):
        #     self._add_input(self.params["input2_idx"], self.params["input2_dtype"], self.params["output_size"], 1, 1)
        # self._add_output(self.params["output_idx"], self.params["output_dtype"], self.params["output_size"], 1, 1)
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

    def get_macs(self):
        p = self.params
        return p["input_size"]
    
    # Shiming: for MUL with constant input
    def get_weights_size(self) -> int:
        p = self.params
        if p["input2_const"] is None: return 0
        if p["input_dtype"] in {"float32", "fp32"}:
            size = 4
        else:
            size = 1
        return p["input2_h"] * p["input2_w"] * p["input2_c"] * size

    def generate_inference_str(self):
        params = self.params

        if params["input_dtype"] == "float32":
            if self.params["input_size"] != self.params["input2_size"]:
                if not islabelstr(self.params["input_idx"]):
                    input0_ptr = f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])}"
                else:
                    input0_ptr = "labels"
                if isParamstr(self.params["input2_idx"]):
                    if "add" not in self.params["input2_idx"] and "scale" in self.params["input2_idx"]:
                        input2_ptr = f"scales{self.params['scale_conv_2d_op'].params['parsed_trainable']}"
                    else:
                        input2_ptr = None  # we don't
                elif not islabelstr(self.params["input2_idx"]):
                    input2_ptr = f"{self._getBufferstr(params['input2_buf_add'], params['input2_buf_add_offset'])}"
                else:
                    input2_ptr = "labels"

                if self.params["input_size"] > self.params["input2_size"]:
                    input_array_ptr = input0_ptr
                    scaler = input2_ptr
                    input_size = self.params["input_size"]
                    scaler_size = self.params["input2_size"]
                else:
                    input_array_ptr = input2_ptr
                    scaler = input0_ptr
                    input_size = self.params["input2_size"]
                    scaler_size = self.params["input_size"]

                if scaler_size > 1:
                    # we need loop over HW dimensions
                    HW_cout = int(input_size / scaler_size)
                    assert HW_cout > 1
                    if self.params["inplace"]:
                        string = (
                            f"fptr = {input_array_ptr};\n"
                            + f"fptr2 = {scaler};\n"
                            + f"for(int hw = 0; hw < {HW_cout}; hw++)"
                            + "{\n"
                            + (
                                f"for(int i = 0; i < {scaler_size}; i++)"
                                + "{float f = *fptr; *fptr++ = fptr2[i] * f;};\n"
                            )
                            + "}\n"
                        )
                    else:
                        string = (
                            f"fptr = {input_array_ptr};\n"
                            + "fptr3 = (float*)"
                            + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])};"
                            + f"fptr2 = {scaler};\n"
                            + f"for(int hw = 0; hw < {HW_cout}; hw++)"
                            + "{\n"
                            + f"for(int i = 0; i < {scaler_size}; i++) *fptr3++ = fptr2[i] * *fptr++;\n"
                            + "}\n"
                        )
                else:
                    string = f"fptr = (float*){input_array_ptr};"
                    string += (
                        "fptr3 = (float*)"
                        + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])};"
                    )
                    # if it is from parameter
                    if self.params["scale_from_add"] is not None:
                        string += (
                            f"for(int i = 0; i < {self.params['output_size']}; i++) fptr3[i] = "
                            + f"{self.params['scale_from_add']} * fptr[i];\n"
                        )
                    elif isinstance(self.params["constant"], float):
                        string += (
                            f"for(int i = 0; i < {self.params['output_size']}; i++) fptr3[i] = "
                            + f"{self.params['constant']} * fptr[i];\n"
                        )
                    else:
                        string += f"fptr2 = {scaler};"
                        string += (
                            f"for(int i = 0; i < {self.params['output_size']}; i++) fptr3[i] = *fptr2 * fptr[i];\n"
                        )
            else:
                if isParamstr(self.params["input2_idx"]):
                    assert self.params["scale_conv_2d_op"] is not None
                    string = (
                        f"mul({self.params['output_size']},"
                        + f"{self._getBufferstrCast(params['input_buf_add'], params['input_buf_add_offset'])},"
                        + f"scales{self.params['scale_conv_2d_op'].params['parsed_trainable']},"
                        + f"{self._getBufferstrCast(params['output_buf_add'], params['output_buf_add_offset'])});\n"
                    )
                elif islabelstr(self.params["input2_idx"]):
                    string = (
                        f"mul({self.params['output_size']},"
                        + f"{self._getBufferstrCast(params['input_buf_add'], params['input_buf_add_offset'])},"
                        + "labels,"
                        + f"{self._getBufferstrCast(params['output_buf_add'], params['output_buf_add_offset'])});\n"
                    )
                else:
                    string = (
                        f"mul({self.params['output_size']},"
                        + f"{self._getBufferstrCast(params['input_buf_add'], params['input_buf_add_offset'])},"
                        + f"{self._getBufferstrCast(params['input2_buf_add'], params['input2_buf_add_offset'])},"
                        + f"{self._getBufferstrCast(params['output_buf_add'], params['output_buf_add_offset'])});\n"
                    )
        else:
            # Shiming:
            string = ""
            if params["broadcast"]:
                if params["broadcast_on_axis"] == [1, 2]:
                    string = (
                        f"mul_broadcast_axis_1_2({params['input_h']}, {params['input_w']}, {params['input_c']}"
                        + f", {params['input_zero_point'] * -1}, {params['input2_zero_point'] * -1}, {params['output_zero_point']}"
                        + f", {params['output_multiplier']}, {params['output_shift']}"
                        + f", {self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                    )
                    if params["input2_const"] is not None:
                        string += f"constinput_{params['parsed_trainable']},"
                    else:
                        string += f"{self._getBufferstr(params['input2_buf_add'], params['input2_buf_add_offset'])},"
                    string += (
                        f"{params['output_activation_min']}, {params['output_activation_max']}"
                        + f", {self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
                    )
                else: raise NotImplementedError
            else:
                # Shiming: Haven't implemented this...
                raise NotImplementedError

        return string
