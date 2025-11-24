# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   TfliteConvertor.py
#
# Reference papers:
#  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
#  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
#  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
# Contact authors:
#  - Wei-Ming Chen, wmchen@mit.edu
#  - Wei-Chen Wang, wweichen@mit.edu
#  - Ji Lin, jilin@mit.edu
#  - Ligeng Zhu, ligeng@mit.edu
#  - Song Han, songhan@mit.edu
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

import logging

import code_generator.converters.tflite_parser as TF_Parser
from code_generator.converters.tflite_parser.mean1dto2d import MEAN2D
from code_generator.converters.tflite_parser.utils import get_input_tensors, get_output_tensors, getOpCodeStr

from .constant import SKIP_OPs
from .tflite import Model


# Parse tflite model into TinyEngine IR format
class TfliteConvertor(object):
    def __init__(self, filepath):
        # path to the tflite file
        self.filepath = filepath
        self.model = self.loadTFmodel(filepath)
        self.subgraph = self.model.Subgraphs(0)
        self.layer = []
        self.tmpPADIndice = None
        self.skip_transpose = None
        self.average_1D_to_2D_holder = MEAN2D()  # For merging 1D to 2D

    # public functions
    def loadTFmodel(self, filepath):
        buf = open(filepath, "rb").read()
        return Model.Model.GetRootAsModel(buf, 0)

    def dumpModelInfo(self):
        version = self.model.Version()
        print("Model version:", version)
        description = self.model.Description().decode("utf-8")
        print("Description:", description)
        subgraph_len = self.model.SubgraphsLength()
        print("Subgraph length:", subgraph_len)

        self.dumpLayerInfo()

    def dumpLayerInfo(self):
        print("Layer length:", len(self.layer))

        # print brief info about each layer
        for i, layer in enumerate(self.layer):
            if self.layer[i]["op"] == "ADD":
                print(
                    "op:",
                    layer["op"],
                    ",input_idx:",
                    layer["input_idx"],
                    ",input2_idx:",
                    layer["input2_idx"],
                    "output_idx:",
                    layer["output_idx"],
                )
            else:
                print(
                    "op:",
                    layer["op"],
                    ",input_idx:",
                    layer["input_idx"],
                    "output_idx:",
                    layer["output_idx"],
                )

    def parseOperatorInfo(self):
        operators_len = self.subgraph.OperatorsLength()

        skip_next_ops = 0
        for i in range(operators_len):
            if skip_next_ops > 0:
                skip_next_ops -= 1
                continue

            op = self.subgraph.Operators(i)
            if i + 2 < operators_len - 2:
                next_op = self.subgraph.Operators(i + 1)
                next_next_op = self.subgraph.Operators(i + 2)
                three_op_sequence = [op, next_op, next_next_op]

                if self.checkIfRequireSEelementmult(three_op_sequence):
                    logging.info("found SE block")
                    skip_next_ops = 2
                    #         -> MEAN -> MEAN -> PWCONV -> PWCONV -> | ADD -> MUL ->     |
                    #  DWCONV                                        |            -> MUL |
                    #                                                |   SEelementmult   |
                    SEelementmult_op = TF_Parser.parse_SEelement(three_op_sequence, self.model, self.layer)

                    self.layer.append(SEelementmult_op)
                    continue

            # parse the op
            self._handleOperator(op)

    # handle one op and parse it into layers[] for supported operators
    def _handleOperator(self, op):
        op_code_str = getOpCodeStr(op, self.model)
        if op_code_str == "CONV_2D":
            self.layer.append(TF_Parser.parse_conv2d(op, self.model, self.tmpPADIndice))
            self.tmpPADIndice = None
        elif op_code_str == "ADD":
            self.layer.append(TF_Parser.parse_add(op, self.model))
        elif op_code_str == "AVERAGE_POOL_2D":
            self.layer.append(TF_Parser.parse_avgpool(op, self.model))
        elif op_code_str == "DEPTHWISE_CONV_2D":
            self.layer.append(TF_Parser.parse_conv2d(op, self.model, self.tmpPADIndice))
            self.tmpPADIndice = None
        elif op_code_str == "PAD":
            self._convert_PAD(op)
        elif op_code_str == "RESIZE_NEAREST_NEIGHBOR":
            self.layer.append(TF_Parser.parse_upsample(op, self.model))
        elif op_code_str == "MAX_POOL_2D":
            self.layer.append(TF_Parser.parse_maxpool(op, self.model, self.tmpPADIndice))
            # Shiming: allow merging previous padding layer with max_pool
            self.tmpPADIndice = None
        elif op_code_str in "MEAN":
            # Shiming: using my own MEAN
            # ret_op = TF_Parser.parse_mead1dto2d(op, self.model, self.average_1D_to_2D_holder)
            # if ret_op is not None:
            #     # TODO: This only handle a specific graph: TRANSPOSE -> MEAN -> MEANS
            #     if self.skip_transpose is not None:
            #         ret_op.params["input_idx"] = self.skip_transpose.input_idx
            #         ret_op.input_tensors[0].graph_idx = self.skip_transpose.input_idx
            #     self.layer.append(ret_op)
            self.layer.append(TF_Parser.parse_reduce(op, self.model, "mean"))
        elif op_code_str == "TRANSPOSE":
            # Shiming: we need to implement this op
            self.layer.append(TF_Parser.parse_transpose(op, self.model))
            # self._convert_TRANSPOSE(op)
        elif op_code_str == "FULLY_CONNECTED":
            self.layer.append(TF_Parser.parse_fc(op, self.model))
        # Shiming:
        elif op_code_str == "QUANTIZE":
            self.layer.append(TF_Parser.parse_quantize(op, self.model))
        elif op_code_str == "RESHAPE":
            ret_op = TF_Parser.parse_reshape(op, self.model)
            if ret_op.params["keep_length"]:
                # Hard-coded: skip tensor connection to RESHAPE layers
                self._remove_RESHAPE(ret_op)
                # self.layer[-1].change_output_tensor_idx(
                #     ret_op.output_tensors[0].graph_idx
                # )
            else:
                raise NotImplementedError(f"RESHAPE layer tries to change "
                                          "total tensor length")
        elif op_code_str == "SOFTMAX":
            self.layer.append(TF_Parser.parse_softmax(op, self.model))
        elif op_code_str == "REDUCE_MAX":
            self.layer.append(TF_Parser.parse_reduce(op, self.model, "max"))
        elif op_code_str == "SHAPE":
            self.layer.append(TF_Parser.parse_shape(op, self.model))
        elif op_code_str == "STRIDED_SLICE":
            self.layer.append(TF_Parser.parse_strided_slice(op, self.model))
        elif op_code_str == "PACK":
            self.layer.append(TF_Parser.parse_pack(op, self.model))
        elif op_code_str == "MUL":
            # Need to support MUL...
            self.layer.append(TF_Parser.parse_mul(op, self.model))
        elif op_code_str in SKIP_OPs:
            pass
        else:
            raise NotImplementedError(f"Unsupported {op_code_str}")

    #         -> MEAN -> MEAN -> PWCONV -> PWCONV -> | ADD -> MUL ->     |
    #  DWCONV                                        |            -> MUL |
    #                                                |    Fuse Target    |
    def checkIfRequireSEelementmult(self, three_op_sequence):
        if (
            getOpCodeStr(three_op_sequence[0], self.model) == "ADD"
            and getOpCodeStr(three_op_sequence[1], self.model) == "MUL"
            and getOpCodeStr(three_op_sequence[2], self.model) == "MUL"
        ):
            return True
        return False

    def _convert_PAD(self, op):
        # get input, weight, and output tensors
        input_tensors = get_input_tensors(op, self.model)
        input_tensor = input_tensors[0]
        paddings = input_tensors[1]

        output_tensors = get_output_tensors(op, self.model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # fuse pad into conv
        self.tmpPADIndice = PAD_tensorIndice(
            input_tensor.tensor_idx, output_tensor.tensor_idx,
            input_tensor, paddings
        )

    def _convert_TRANSPOSE(self, op):
        # get input, weight, and output tensors
        input_tensors = get_input_tensors(op, self.model)
        input_tensor = input_tensors[0]

        output_tensors = get_output_tensors(op, self.model)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # fuse pad into conv
        self.skip_transpose = PAD_tensorIndice(input_tensor.tensor_idx, output_tensor.tensor_idx)

    def _remove_RESHAPE(self, op):
        # Delete layer, since data are placed continuously in the buffer anyway
        # Skip data connection: last layer just outputs to SHAPE's output
        keep_searching = True
        while keep_searching:
            keep_searching = False
            for l in self.layer:
                if l.output_tensors[0].graph_idx \
                            == op.input_tensors[0].graph_idx:
                    print("Connecting {:s}'s output to RESHAPE's output".format(l.params["op"]))
                    l.change_output_tensor_idx(
                        op.output_tensors[0].graph_idx
                    )
                    keep_searching = True
        # Delete shape tensor computation, if any
        keep_searching = True
        useless_tensor_idx = [str(op.params["shape_idx"])]
        while keep_searching:
            keep_searching = False
            to_del = None
            for idx, l in enumerate(self.layer):
                if (len(l.output_tensors) == 1) and \
                        (l.output_tensors[0].graph_idx in useless_tensor_idx):
                    for in_tensor in l.input_tensors:
                        useless_tensor_idx.append(in_tensor.graph_idx)
                    to_del = idx
                    break
            if to_del is not None:
                print("Deleting parsed layer {:d}: {:s}".format(to_del, self.layer[to_del].params["op"]))
                del self.layer[to_del]
                keep_searching = True



class PAD_tensorIndice(object):
    def __init__(self, input_idx, output_idx, last_input_tensor, paddings):
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.last_input_tensor = last_input_tensor
        self.paddings = paddings
