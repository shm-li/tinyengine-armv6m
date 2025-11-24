import warnings

from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

default_params = {
    # op related
    "op": "SOFTMAX",
    "input_idx": None,
    "output_idx": None,
    # tensor related
    "input_idx": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "output_idx": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "input_dtype": None,
    "output_dtype": None, 
    # quantization related
    "input_zero_point": None,
    "input_scale": None,
    "output_zero_point": None,
    "output_scale": None,
    # Softmax related
    "beta": None,
    "input_multiplier": None,
    "input_left_shift": None,
    "diff_min": None,
}


class Softmax(basicOperator):
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
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )
        if None in default_params:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def generate_inference_str(self):
        string = ""
        params = self.params
        if params["input_dtype"] == "float32":
            raise NotImplementedError("No support for f32 input in SOFTMAX")
        else:
            string += (
                f"softmax_int8"
                + f"({self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                + f"{str(int(params['input_h']*params['input_w']*params['input_c']))}, "
                + f"{params['input_multiplier']},{params['input_left_shift']},{params['diff_min']}, "
                + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
            )
        return string
