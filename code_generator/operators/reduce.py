import warnings

from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

default_params = {
    # op related
    "op": "REDUCE",
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
    # reduce related
    "reduce_type": None,
    "keep_dims": None,
    "axis": None,
    # quantization related (for mean)
    "input_zero_point": None,
    "output_scale": None,
    "output_zero_point": None,
    "multiplier": None,
    "shift": None,
    "max_val": None,
    "min_val": None,
}


class Reduce(basicOperator):
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
        axis_str = "_".join([str(each) for each in params["axis"]])
        if axis_str != "1_2":
            raise NotImplementedError("Axis {:s} not supported in "
                                      "reduce layer".format(axis_str))
        if params["reduce_type"] == "max":
            string += (
                f"reduce_{params['reduce_type']}_axis_{axis_str}_{params['input_dtype']}"
                + f"({self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                + f"{params['input_h']},{params['input_w']},{params['input_c']}, "
                + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
            )
        elif params["reduce_type"] == "mean":
            string += (
                f"reduce_{params['reduce_type']}_axis_{axis_str}_{params['input_dtype']}"
                + f"({self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                + f"{params['input_h']},{params['input_w']},{params['input_c']}, "
                + f"{params['multiplier']},{params['shift']},"
                + f"{params['input_zero_point']}, {params['output_zero_point']}, "
                + f"{params['min_val']},{params['max_val']}, "
                + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
            )
        return string
