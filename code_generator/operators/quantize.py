import warnings

from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["Quantize"]

default_params = {
    # op related
    "op": "QUANTIZE",
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
    "input_zero_point": None,
    "input_scale": None,
    "input_dtype": None,
    "output_zero_point": None,
    "output_scale": None,
    "output_dtype": None,
    # quantization related
    "requantize": False,
    "min_val": None,
    "max_val": None,
    "multiplier": None,
    "shift": None,
}


class Quantize(basicOperator):
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
        # print(self.params["input_dtype"], self.params["output_dtype"])
        # print(params["input_dtype"], params["output_dtype"])
        # print(self.input_tensors[0].dtype)
        # print(self.output_tensors[0].dtype)

        if None in default_params:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def generate_inference_str(self):
        string = ""
        params = self.params
        if params["input_dtype"] == "float32":
            # Do quantization
            string += (
                f"quantize_{params['input_dtype']}_to_{params['output_dtype']}"
                + f"({self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                + f"{str(int(params['input_h']*params['input_w']*params['input_c']))}, "
                + f"{str(params['output_scale'])},{str(params['output_zero_point'])},"
                + f"{str(params['min_val'])},{str(params['max_val'])},"
                + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
            )
        else:
            # Do re-quantization
            string += (
                f"requantize_{params['input_dtype']}_to_{params['output_dtype']}"
                + f"({self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                + f"{str(int(params['input_h']*params['input_w']*params['input_c']))}, "
                + f"{str(params['multiplier'])},{str(params['shift'])},"
                + f"{str(params['input_zero_point'])},{str(params['output_zero_point'])},"
                + f"{str(params['min_val'])},{str(params['max_val'])},"
                + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
            )
        return string
