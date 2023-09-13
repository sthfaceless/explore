import os
import textwrap
from collections import defaultdict
from enum import Enum
from itertools import groupby
import glob
import cv2
import yaml

from math import ceil
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

np.set_printoptions(threshold=4096)
import json
from copy import deepcopy, copy
from scipy.signal import convolve
import yaml

import onnx
import onnxruntime as ort

cfg = yaml.load(open('config.yaml', 'r'), yaml.FullLoader)

MODEL_ONNX = '../reduce.onnx'
OUTPUT_FILE = 'D:\\PycharmProjects\\explore\\MAGPIE\\effects\\reduce.hlsl'
HEADER_FILE = cfg['header']

MAGPIE = cfg['target']['magpie']
VULKAN = cfg['target']['vulkan']

BASE_TYPE = cfg['base_type']
BLOCK_SIZE = cfg['block_size']

# merge ops from previous pass to reduce write ops but increase computation ops
MERGE_BACK = cfg['heuristics']['merge_back']
# matrix mul vectorization
VECTORIZED = cfg['heuristics']['vectorized']
DOT_VECTORIZED = cfg['heuristics']['dot_vectorized']
# whether upscale/downscale operation requires new pass (may speedup shader with removing redundant calculations)
SYNC_UPSCALE = cfg['heuristics']['sync_upscale']
SYNC_DOWNSCALE = cfg['heuristics']['sync_downscale']
# onnx names which requires new pass (Conv_10, Slice_4 ... )
SYNC_NAMES = cfg['heuristics']['sync_names']

model = onnx.load(MODEL_ONNX)
onnx.checker.check_model(model)
onnx_graph = model.graph
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'

DEBUG = False

# with open('reduce.graph', 'w') as f:
#     f.write(str(onnx_graph))

# UNORM | SNORM | None
QUANTIZE_RW = None
SCALE = 2
ACT_ONNX = MODEL_ONNX.replace('.onnx', '_act.onnx')
ACT_OUT = MODEL_ONNX.replace('.onnx', '.json')
IMAGES_ROOT = '../images/hr'


def get_YUV(R, G, B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 0.5 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 0.5 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr


def convert_RGB2YUV(image):
    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    return np.stack(get_YUV(R, G, B), axis=-1)


def to_array(img):
    return np.moveaxis(convert_RGB2YUV(img.astype(np.float32) / 255), -1, 0)[None,]


def np2d(arr):
    assert len(arr.shape) == 2
    return '\n'.join(', '.join(row.astype(str).tolist()) for row in arr)


if QUANTIZE_RW:

    qmodel = onnx.load(MODEL_ONNX)
    onnx.checker.check_model(qmodel)

    names = [node.output[0] for node in qmodel.graph.node
             if node.op_type not in ('Constant', 'ConstantOfShape') and node.output[0] != 'output']

    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(qmodel)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name in names:
            value_info_protos.append(node)
    assert len(value_info_protos) == len(names)
    names.insert(0, 'output')

    # in inference stage, these tensor will be added to output dict.
    qmodel.graph.output.extend(value_info_protos)
    onnx.checker.check_model(qmodel)

    onnx.save(qmodel, ACT_ONNX)

    lr_shape = tuple(item.dim_value for item in model.graph.input[0].type.tensor_type.shape.dim)[-2:][::-1]
    hr_shape = tuple(dim * SCALE for dim in lr_shape)
    lrs, hrs = [], []
    for path in glob.glob(os.path.join(IMAGES_ROOT, '*.png')):
        hr = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        hrs.append(cv2.resize(hr, hr_shape, interpolation=cv2.INTER_LINEAR))
        lrs.append(to_array(cv2.resize(hr, lr_shape, interpolation=cv2.INTER_LINEAR)))

    ort_session = ort.InferenceSession(ACT_ONNX)
    outputs = [ort_session.run(None, {"input": lr[:, :1]}) for lr in lrs]


    def calc_stats(item):
        item = item.flatten()
        stats = {
            'mean': item.mean(),
            'std': item.std(),
            'min': item.min(),
            'max': item.max(),
            '01': np.quantile(item, 0.001),
            '99': np.quantile(item, 0.999),
        }
        return {k: float(v) for k, v in stats.items()}


    activations = {
        name: calc_stats(np.stack([out[idx] for out in outputs], axis=0))
        for idx, name in enumerate(names)
    }

    with open(ACT_OUT, 'w') as f:
        json.dump(activations, f)
    os.remove(ACT_ONNX)

    with open(ACT_OUT, 'r') as f:
        ACTIVATIONS = json.load(f)


def plot_graph(ops):
    G = nx.DiGraph()
    for op in ops:
        for inp in op.input:
            G.add_edge(inp, op.output)

    from pylab import rcParams
    rcParams['figure.figsize'] = 20, 20
    pos = nx.spring_layout(G, scale=20, k=3 / np.sqrt(G.order()))
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True)
    plt.show()


def allocate_priority(ops, op, sgn):
    priority = op.priority + sgn
    priors = {item.priority: item for item in ops.values()}

    curr_priority = priority
    while curr_priority in priors:
        curr_op = priors[curr_priority]
        curr_op.priority += sgn
        if curr_op.out_item is not None:
            curr_op.out_item.op_idx = curr_op.priority
        curr_priority = curr_op.priority

    return priority


def allocate_after(ops, op):
    return allocate_priority(ops, op, 1)


def allocate_before(ops, op):
    return allocate_priority(ops, op, -1)


def default_scale(scale_change):
    return all(abs(scale - 1) < 1e-6 for scale in scale_change)


def get_min_scale(scales):
    return tuple(min(scale[dim_idx] for scale in scales) for dim_idx in range(len(scales[0])))


def round4(val):
    return val if val % 4 == 0 else val + 4 - (val % 4)


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def dtype_mapper(idx):
    if idx == 1:
        return np.float32
    elif idx == 2:
        return np.uint8
    elif idx == 3:
        return np.int8
    elif idx == 7:
        return np.int64
    elif idx == 9:
        return np.bool
    elif idx == 10:
        return np.float16


def get_scale_key(scale, multiplier=10):
    if all(scale[0] == val for val in scale):
        return f"{int(scale[0] * multiplier):02d}"
    else:
        return ''.join([f"{int(val) * multiplier:02d}" for val in scale])


def prec(val, tp='float'):
    if 'float' in tp:
        suffix = 'f'
    elif 'half' in tp:
        suffix = 'h'
    else:
        suffix = ''
    return f'{val:.6f}{suffix}'


def type_size(tp):
    if ord('1') <= ord(tp[-1]) <= ord('9'):
        return int(tp[-1])
    else:
        return 1


def add_between(op, first, second):
    for idx in [idx for idx, item in enumerate(first.output) if item == second]:
        first.output[idx] = op
    for idx in [idx for idx, item in enumerate(second.input) if item == first]:
        second.input[idx] = op
    op.input.append(first)
    op.output.append(second)


def merge_list_dicts(*dicts):
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    result = defaultdict(list)
    for k in keys:
        for d in dicts:
            result[k] += d[k]
    return result


def concat_variables(prev_vars, variables):
    var_sizes = list(reversed([[type_size(var[1]), var[1]] for var in variables]))
    values = []
    for inp_var, inp_tp in prev_vars:
        inp_size = type_size(inp_tp)
        var_size, var_tp = var_sizes[-1]
        if (len(values) == 0 or isinstance(values[-1], str)) and var_size == inp_size:
            values.append(inp_var)
            var_sizes.pop()
        else:
            if len(values) == 0 or isinstance(values[-1], str):
                values.append([])
            min_size = min(inp_size, var_size)
            values[-1].extend([f"{inp_var}{'.' + 'xyzw'[idx] if inp_size > 1 else ''}"
                               for idx in range(min_size)])
            var_sizes[-1][0] -= min_size
            if var_sizes[-1][0] == 0:
                values[-1] = f"{var_tp}({', '.join(values[-1])})"
                var_sizes.pop()
            if inp_size - min_size > 0:
                values.append([f"{inp_var}.{'xyzw'[idx]}" for idx in range(min_size, inp_size)])
                var_sizes[-1][0] -= len(values[-1])
                if var_sizes[-1][0] == 0:
                    values[-1] = f"{var_sizes[-1][1]}({', '.join(values[-1])})"
                    var_sizes.pop()
    return values


def apply_op(expr, op, var_idx, tp, step, fun_defines):
    op_signs = {
        'Add': '+',
        'Mul': '*',
        'Div': '/',
        'Sub': '-'
    }
    if op.op_type in op_signs:
        return f"({f' {op_signs[op.op_type]} '.join([expr] + [prec(constant.w, tp) for constant in op.constants])})"
    elif op.op_type.lower() in fun_defines:
        return f'{op.op_type.upper()}({expr})'
    elif op.op_type == 'Clip':
        return f"min(max({expr}, {prec(op.clip_min, tp)}), {prec(op.clip_max, tp)})"
    elif op.op_type == 'LeakyRelu':
        return f"(max(0, {expr}) + {prec(op.slope, tp)} * min(0, {expr}))"
    elif op.op_type == 'PRelu':
        if len(op.slope) == 1:
            value = prec(op.slope[0], tp)
        else:
            value = f"{tp}({', '.join([prec(item, tp) for item in op.slope[var_idx: var_idx + step]])})"
        return f"(max(0, {expr}) + {value} * min(0, {expr}))"
    else:
        raise NotImplementedError(f'Not implemented applying operation of type {op.op_type}')


def matrix_mul(var, tp, inp_var, inp_tp, matrix):
    code = ""
    if VECTORIZED:
        matrix_values = ',\n\t'.join(', '.join(prec(val, tp) for val in row) for row in matrix)
        code += f"{var} += mul({tp if type_size(tp) > 1 else f'{tp}1'}x{type_size(inp_tp)}(\n\t" \
                f"{matrix_values}" \
                f"), {inp_var});\n"
    else:
        for local_idx in range(type_size(tp)):
            var_part = var if type_size(tp) == 1 else var + '.' + 'xyzw'[local_idx]
            if DOT_VECTORIZED:
                values = ', '.join(prec(val, inp_tp) for val in matrix[local_idx])
                code += f"{var_part} += dot({inp_tp}({values}), {inp_var});\n"
            else:
                values = ' + '.join(
                    f"{inp_var}{'.' + 'xyzw'[val_idx] if type_size(inp_tp) > 1 else ''} * {prec(val, inp_tp)}"
                    for val_idx, val in enumerate(matrix[local_idx]))
                code += f"{var_part} += {values};\n"

    return code


class ShaderManager:

    def __init__(self, header='header.hlsl', block_size=8):
        self.block_size = block_size
        self.num_threads = block_size ** 2
        self.inputs = defaultdict(set)
        self.outputs = defaultdict(set)
        self.passes = defaultdict(list)

        # variables for compiling
        self.result = []
        self.pass_inputs = defaultdict(set)
        self.pass_outputs = defaultdict(set)
        self.texs = set()
        self.scales = dict()

        # process header
        assert os.path.exists(header), f'Header {header} does not exists!'
        with open(header, 'r') as f:
            self.header_lines = f.readlines()
        self.header = ''.join(self.header_lines)
        if VULKAN:
            self.header = self.header.replace('GetInputSize()', 'float2(input_tex_size.w, input_tex_size.h)')

        self.defines = set([line for line in self.header_lines if line.startswith('#define')])
        self.fun_defines = set([line.split()[1].split('(')[0].lower() for line in self.defines
                                if line.split()[1].endswith('(x)')])
        self.sampler = [line.split()[1][:-1] for line in self.header_lines if 'SamplerState' in line][0]

    def add_write_tex(self, tex, depth):
        self.outputs[depth].add(tex)

    def add_read_tex(self, tex, depth):
        self.inputs[depth].add(tex)

    def __add_text(self, text):
        self.result.extend(text.split('\n'))

    def __add_lines(self, lines):
        self.result.extend(lines)

    def add_pass_text(self, text, depth):
        self.passes[depth].extend(text.split('\n'))

    def prepare_head(self):
        self.__add_text(self.__get_header_tokens('//!HEAD'))

    def prepare_common(self):
        self.__add_text("//!COMMON")
        self.__add_text(self.__get_header_tokens('//!COMMON'))

    def prepare_textures(self):
        # get real names for all input/output textures
        self.texs, self.scales = set(), dict()
        for pass_names, pass_objs in ((self.pass_inputs, self.inputs), (self.pass_outputs, self.outputs)):
            for depth, tex_items in pass_objs.items():
                for tex_item in tex_items:
                    tex_names = tex_item.get_names()
                    pass_names[depth].update(tex_names)
                    self.scales.update({name: tex_item.scale for name in tex_names})
                self.texs.update(pass_names[depth])
                pass_names[depth] = list(sorted(pass_names[depth]))

        # remove input/output textures from declaration as they already declared by default
        self.texs.discard('INPUT')
        self.texs.discard('OUTPUT')

        # declare all textures
        for tex_id, name in enumerate(sorted(self.texs,
                                             key=lambda name: (name.split('_')[-2], int(name.split('_')[-1])))):
            self.__add_text(textwrap.dedent(f"""
            //!TEXTURE
            //!WIDTH INPUT_WIDTH {f'* {self.scales[name][1]}' if self.scales[name][1] != 1 else ''}
            //!HEIGHT INPUT_HEIGHT {f'* {self.scales[name][0]}' if self.scales[name][0] != 1 else ''}
            //!FORMAT {f'R8G8B8A8_{QUANTIZE_RW.upper()}' if QUANTIZE_RW else 'R16G16B16A16_FLOAT'}
            Texture2D {name};
            """))

    def __get_header_tokens(self, key):
        tokens = self.header.split(key)
        if len(tokens) > 1:
            return tokens[1].strip()
        else:
            return ''

    def prepare_pass(self, pass_id):
        pass_header = textwrap.dedent(f"""
        //!PASS {pass_id + 1}
        //!DESC L1: Pass_{pass_id + 1}
        //!IN {', '.join(self.pass_inputs[pass_id])}
        //!OUT {', '.join(self.pass_outputs[pass_id])}
        //!BLOCK_SIZE {self.block_size}
        //!NUM_THREADS {self.num_threads}
        """)
        if self.pass_outputs[pass_id][0] == 'OUTPUT':
            pass_header = '\n'.join([line for line in pass_header.split('\n') if '!OUT' not in line])

        if VULKAN:
            pass_header += self.__get_header_tokens('//!VULKAN')
        if pass_id == min(self.passes.keys()):
            pass_header += self.__get_header_tokens('//!BEGIN')
        if pass_id == max(self.passes.keys()):
            pass_header += self.__get_header_tokens('//!END')

        pass_header += textwrap.dedent(f"""
            void Pass{pass_id + 1}(uint2 blockStart, uint3 threadId){{\n
                uint2 pos = blockStart + uint2(threadId.x / {self.block_size}, threadId.x % {self.block_size});
        """)
        self.__add_text(pass_header)
        self.__add_lines([f'\t{line}' for line in self.passes[pass_id]])
        self.__add_text('}')

    def post_process(self):
        result = []
        for line in self.result:
            if '//!OUTPUT_WIDTH' in line:
                output_scale = self.scales['OUTPUT']
                line = line.replace('INPUT_WIDTH',
                                    f'INPUT_WIDTH{f" * {output_scale[1]}" if output_scale[1] != 1 else ""}')
            if '//!OUTPUT_HEIGHT' in line:
                output_scale = self.scales['OUTPUT']
                line = line.replace('INPUT_HEIGHT',
                                    f'INPUT_HEIGHT{f" * {output_scale[0]}" if output_scale[0] != 1 else ""}')
            result.append(line)
        self.result = result

    def clear(self):
        self.result.clear()
        self.pass_inputs.clear()
        self.pass_outputs.clear()
        self.scales.clear()
        self.texs.clear()

    def compile(self):
        self.clear()
        self.prepare_head()
        self.prepare_textures()
        self.prepare_common()

        for pass_id in self.passes.keys():
            self.prepare_pass(pass_id)

        self.post_process()

        return '\n'.join(self.result)


class TextureManager:

    def __init__(self):
        self.textures = {}

    def get_textures(self, channels, depth, scale):
        # get list of textures for this scale
        scale_key = get_scale_key(scale)
        if scale_key not in self.textures:
            self.textures[scale_key] = []

        # take only texs that already used
        texs = [tex for tex in self.textures[scale_key] if tex.max_requirement() >= depth]

        # greedy find place for new textures
        if len(texs) == 0:
            begin, end = 0, channels - 1
        else:
            for tex1, tex2 in zip([Texture(begin=-1, end=-1)] + texs, texs + [Texture(begin=1 << 31, end=1 << 31)]):
                start = tex1.end + 1
                start = start if start % 4 == 0 else start + 4 - (start % 4)
                if channels <= tex2.begin - start:
                    begin, end = start, start + channels - 1
                    break
        item = Texture(begin, end, scale)
        texs.append(item)
        self.textures[scale_key] = list(sorted(texs, key=lambda tex: tex.begin))
        return item


class Operation:
    op_type = None

    def __init__(self, name=None, op_type=None, input_names=(), weights=None, depth=0, priority=0, inp=None,
                 scale=(1.0, 1.0),
                 output=None, out_channels=None, out_item=None, onnx_node=None, scale_change=(1.0, 1.0), constants=None,
                 pattern_ops=None, sync=False, requires_shape=(1, 1), read_type=None, **kwargs):

        if self.op_type is None:
            self.op_type = op_type
        assert self.op_type is not None, f'Got None op_type for generic operation {name}'

        self.name = name
        self.onnx_name = onnx_node.name if onnx_node else None
        self.input_names = [name for name in input_names if len(name) > 0]

        # input/output ops
        self.input = inp if inp else list()
        self.output = output if output else list()

        # input constants for operation that was onnx nodes
        self.constants = constants if constants else list()

        # pass idx
        self.depth = depth
        # operations order, first --- 0
        self.priority = priority
        # current scale of operation output relative to input texture
        self.scale = scale
        # does this op requires synchronization
        self.sync = sync
        # rectangle for convolution like operations i.e. (3, 3)
        self.requires_shape = requires_shape
        self.read_shape = kwargs.get('read_shape', Shape.create_mask(self.requires_shape))
        # read type like usual read/resize read
        self.read_type = read_type
        # does this operation changes scale of texture
        self.scale_change = scale_change
        self.run_on_prev_scale = kwargs.get('run_on_prev_scale', False)

        self.out_channels = out_channels
        # out shape + type of operation
        self.out_item = out_item if out_item else kwargs.get('out_item')

        # List of operations merged for this pattern
        self.pattern_ops = pattern_ops

        # Was the result of operation quantized
        self.quantized = kwargs.get('quantized')

    def get_run_scale(self):
        return tuple(float(scale / scale_change) for scale, scale_change
                     in zip(self.scale, self.scale_change)) if self.run_on_prev_scale else self.scale

    def update_depth(self, depth, pass_scales):
        if depth != self.depth:
            idxs = [idx for idx, scales in enumerate(pass_scales[self.depth])
                    if all(abs(s1 - s2) < 1e-6 for s1, s2 in zip(self.get_run_scale(), scales))]
            if len(idxs) > 0:
                pass_scales[self.depth].pop(idxs[0])
            self.depth = depth
            pass_scales[self.depth].append(self.get_run_scale())

    def get_attribute(self, onnx_node, name):
        return [attr for attr in onnx_node.attribute if attr.name == name][0]

    def get_opts(self):
        return {opt: self.__dict__[opt] for opt in ('depth', 'out_channels')}

    def requires_synchronization(self):
        sync = self.sync
        sync |= self.requires_shape != (1, 1)
        sync |= SYNC_UPSCALE and any(scale > 1.0 for scale in self.scale_change)
        sync |= SYNC_DOWNSCALE and any(scale < 1.0 for scale in self.scale_change)
        sync |= self.onnx_name in SYNC_NAMES
        return sync

    def __hash__(self):
        return hash((self.name, self.op_type))

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return False
        return self.name == other.name and self.op_type == other.op_type

    def post_init(self):
        pass

    def _update_channels(self, input_channels):
        if self.name not in INPUT_NAMES:
            assert all([x == input_channels[0] for x in input_channels]), \
                f'Input for {self.name} have different sizes {input_channels}'
            self.out_channels = input_channels[0]

    def update_channels(self):
        input_channels = [op.out_channels for op in self.input if op.op_type not in ('Constant')]
        return self._update_channels(input_channels)

    def process_before(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        for var_idx, _ in enumerate(var_types):
            prev_x = x - center[0] + prev_center[0]
            prev_y = y - center[1] + prev_center[1]
            if prev_x < len(input_exprs[0]) and prev_y < len(input_exprs[0][0]) \
                    and input_exprs[0][prev_x][prev_y] and var_idx < len(input_exprs[0][prev_x][prev_y]):
                out[x][y][var_idx] = input_exprs[0][prev_x][prev_y][var_idx]

    def process_after(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        code = ''
        if len(self.output) > 1 or any(scale > 1 for scale in self.output[0].scale_change) \
                or isinstance(self.output[0], (Conv, ConvTranspose, Repeat, Write)) \
                or isinstance(self, (Concat, Read)):
            for var_idx, (var, var_tp) in enumerate(var_types):
                code += f"{var_tp} {var} = {out[x][y][var_idx][0]};\n"
        else:
            self.out_item.set_exprs(out)
        return code

    def process(self):

        code = f"//{self.op_type} {self.name}\n"
        out_item = self.out_item if isinstance(self.out_item, Vector) else self.input[0].out_item  # for write ops
        mask, out = out_item.shape.mask, out_item.create_exprs()
        variables = out_item.create_exprs()
        input_exprs = [op.out_item.get_exprs() for op in self.input if isinstance(op.out_item, Vector)]

        if DEBUG:
            code += f"/*input\n{np2d(self.read_shape.mask)}*/\n"
            if isinstance(self.out_item, Vector):
                code += f"/*output\n{np2d(self.out_item.shape.mask)}*/\n"

        prev_center = tuple((dim - 1) // 2 for dim in input_exprs[0].shape) if len(input_exprs) > 0 else None
        center = tuple((dim - 1) // 2 for dim in out.shape)

        for x, row in enumerate(variables):
            for y, var_types in enumerate(row):
                if mask[x][y]:
                    self.process_before(x, y, var_types, input_exprs, mask, out, prev_center, center)

        for x, row in enumerate(variables):
            for y, var_types in enumerate(row):
                if mask[x][y]:
                    code += self.process_after(x, y, var_types, input_exprs, mask, out, prev_center, center)

        return code


class Conv(Operation):
    op_type = 'Conv'

    def __init__(self, *args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)

        weights = kwargs.get('weights')
        if weights and self.input_names[-1] in weights and self.input_names[-2] in weights:
            self.w = weights[self.input_names[-2]]
            self.b = weights[self.input_names[-1]]
            self.input_names = self.input_names[:-2]
        else:
            self.w = kwargs.get('w')
            self.b = kwargs.get('b')

        onnx_node = kwargs.get('onnx_node')
        self.groups = self.get_attribute(onnx_node, "group").i if onnx_node else kwargs.get('groups', 1)
        if self.groups > 1:
            assert self.w.shape[0] % self.groups == 0, \
                f'When groups({self.groups}) > 1 for {self.name}' \
                f' out_channels({self.w.shape[0]}) must be divisible to groups'
        self.strides = tuple(self.get_attribute(onnx_node, "strides").ints) if onnx_node else kwargs.get('strides',
                                                                                                         (1, 1))
        if self.strides > (1, 1):
            assert tuple(self.w.shape[-2:]) == self.strides, \
                f'Supporting strided convolution only with same kernel size but got stride {self.strides} and kernel {self.w.shape[-2:]}'
        self.dilations = tuple(self.get_attribute(onnx_node, "dilations").ints) if onnx_node else kwargs.get(
            'dilations', (1, 1))
        if self.dilations > (1, 1):
            assert self.strides == (1, 1), \
                f'Supported dilations only without stride but found ' \
                f'strides {self.strides} and dilations {self.dilations} for {self.name}'

        self.pads = tuple(self.get_attribute(onnx_node, "pads").ints)[:2] if onnx_node else kwargs.get(
            'pads', (1, 1))

        self.scale_change = tuple(1.0 / stride for stride in self.strides)
        self.requires_shape = tuple(self.w.shape[-2:]) if default_scale(self.scale_change) else (1, 1)
        self.read_shape = Shape.create_mask(self.requires_shape, 'conv')

    def _update_channels(self, input_channels):
        assert len(self.input) == 1, \
            f'Get more inputs than 1 for conv operation: {self.name} --- {self.input_names}'
        prev_channels = input_channels[0]
        assert prev_channels // self.groups == self.w.shape[1], \
            f'Input for conv {self.name} --- {prev_channels} mismatch to weight {self.w.shape}'
        self.out_channels = self.w.shape[0]

    def process_after(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        code = ""
        for var_idx, (var, tp) in enumerate(var_types):
            step = min(4, self.out_channels - var_idx * 4)
            values = [prec(val, tp) for val in self.b[4 * var_idx: 4 * var_idx + step]]
            expr = f"{tp}({', '.join(values)})"
            code += f"{tp} {var} = {expr};\n"
            for kx in range(self.w.shape[-2]):
                for ky in range(self.w.shape[-1]):
                    prev_x = int((x - center[0]) / self.scale_change[0] + prev_center[0]
                                 + kx * self.dilations[0] - self.pads[0])
                    prev_y = int((y - center[1]) / self.scale_change[1] + prev_center[1]
                                 + ky * self.dilations[1] - self.pads[1])
                    if self.w.shape[1] * self.groups == self.w.shape[0] and self.groups == self.w.shape[0]:
                        # one to one depthwise convolution
                        matrix = self.w[var_idx * 4: var_idx * 4 + step:, 0, kx, ky]
                        matrix_values = ', '.join([prec(val, tp) for val in matrix])
                        inp_var, inp_tp = input_exprs[0][prev_x][prev_y][var_idx]
                        code += f"{var} += {inp_tp}({matrix_values}) * {inp_var};\n"
                    elif self.groups == 1:
                        for inp_idx, (inp_var, inp_tp) in enumerate(input_exprs[0][prev_x][prev_y]):
                            # usual full convolution
                            inp_step = min(4, self.input[0].out_channels - inp_idx * 4)
                            matrix = self.w[var_idx * 4: var_idx * 4 + step,
                                     inp_idx * 4: inp_idx * 4 + inp_step,
                                     kx, ky]
                            code += matrix_mul(var, tp, inp_var, inp_tp, matrix)
        return code


class ConvTranspose(Operation):
    op_type = 'ConvTranspose'

    def __init__(self, *args, **kwargs):
        super(ConvTranspose, self).__init__(*args, **kwargs)

        weights = kwargs.get('weights')
        if weights and self.input_names[-1] in weights and self.input_names[-2] in weights:
            self.w = weights[self.input_names[-2]]
            self.b = weights[self.input_names[-1]]
            self.input_names = self.input_names[:-2]
        else:
            self.w = kwargs.get('w')
            self.b = kwargs.get('b')

        onnx_node = kwargs.get('onnx_node')
        self.groups = self.get_attribute(onnx_node, "group").i if onnx_node else kwargs.get('groups', 1)

        self.strides = tuple(self.get_attribute(onnx_node, "strides").ints) if onnx_node else kwargs.get('strides',
                                                                                                         (1, 1))
        if self.strides > (1, 1):
            assert tuple(self.w.shape[-2:]) == self.strides, \
                f'Supporting strided convolution only with same kernel size but got stride {self.strides} and kernel {self.w.shape[-2:]}'

        self.dilations = tuple(self.get_attribute(onnx_node, "dilations").ints) if onnx_node else kwargs.get(
            'dilations', (1, 1))
        if self.dilations > (1, 1):
            assert self.strides == (1, 1), \
                f'Supported dilations only without stride but found ' \
                f'strides {self.strides} and dilations {self.dilations} for {self.name}'

        self.scale_change = tuple(float(stride) for stride in self.strides)
        self.run_on_prev_scale = True

    def _update_channels(self, input_channels):
        assert len(self.input) == 1, \
            f'Get more inputs than 1 for conv transpose operation: {self.name} --- {self.input_names}'
        prev_channels = input_channels[0]
        assert prev_channels == self.w.shape[0], \
            f'Input for conv transpose {self.name} --- {prev_channels} mismatch to weight {self.w.shape}'
        self.out_channels = self.w.shape[1] * self.groups

    def process_after(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        code = ""
        for var_idx, (var, tp) in enumerate(var_types):
            step = min(4, self.out_channels - var_idx * 4)
            values = [prec(val, tp) for val in self.b[4 * var_idx: 4 * var_idx + step]]
            code += f"{tp} {var} = {tp}({', '.join(values)});\n"
            prev_x = int((x - center[0]) / self.scale_change[0] + prev_center[0])
            prev_y = int((y - center[1]) / self.scale_change[1] + prev_center[1])
            if self.w.shape[1] * self.groups == self.w.shape[0] and self.groups == self.w.shape[0]:
                # one to one depthwise convolution
                matrix = self.w[var_idx * 4: var_idx * 4 + step:, 0,
                         int((x - center[0]) % self.scale_change[0]),
                         int((y - center[1]) % self.scale_change[1])]
                matrix_values = ', '.join([prec(val, tp) for val in matrix])
                inp_var, inp_tp = input_exprs[0][prev_x][prev_y][var_idx]
                code += f"{var} += {inp_tp}({matrix_values}) * {inp_var};\n"
            elif self.groups == 1:
                for inp_idx, (inp_var, inp_tp) in enumerate(input_exprs[0][prev_x][prev_y]):
                    # usual full convolution
                    inp_step = min(4, self.input[0].out_channels - inp_idx * 4)
                    matrix = self.w[inp_idx * 4: inp_idx * 4 + inp_step, var_idx * 4: var_idx * 4 + step,
                             int((x - center[0]) % self.scale_change[0]),
                             int((y - center[1]) % self.scale_change[1])].transpose()
                    code += matrix_mul(var, tp, inp_var, inp_tp, matrix)
        return code


class Constant(Operation):
    op_type = 'Constant'

    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)

        onnx_node = kwargs.get('onnx_node')
        self.input_names = []
        if onnx_node:
            value = self.get_attribute(onnx_node, "value").t
            w = np.frombuffer(value.raw_data, dtype=dtype_mapper(value.data_type))
            self.w = w[0] if len(value.dims) == 0 else w.reshape(value.dims)
        else:
            self.w = kwargs.get('w')


class ConstantOfShape(Operation):
    op_type = 'ConstantOfShape'

    def __init__(self, *args, **kwargs):
        super(ConstantOfShape, self).__init__(*args, **kwargs)

        onnx_node = kwargs.get('onnx_node')
        self.op_type = 'Constant'
        self.input_names = []
        if onnx_node:
            value = self.get_attribute(onnx_node, "value").t
            w = np.frombuffer(value.raw_data, dtype=dtype_mapper(value.data_type))
            self.w = w[0] if len(value.dims) == 0 else w.reshape(value.dims)
        else:
            self.w = kwargs.get('w')


class DepthToSpace(Operation):
    op_type = 'DepthToSpace'

    def __init__(self, *args, **kwargs):
        super(DepthToSpace, self).__init__(*args, **kwargs)

        onnx_node = kwargs.get('onnx_node')
        self.scale_change = tuple(float(self.get_attribute(onnx_node, "blocksize").i) for _ in range(2)) \
            if onnx_node else kwargs.get('scale_change', (1, 1))
        self.run_on_prev_scale = True

    def _update_channels(self, input_channels):
        assert len(self.input) == 1, \
            f'Get more inputs than 1 for DepthToSpace operation: {self.name} --- {self.input_names}'
        self.out_channels = int(self.input[0].out_channels / (self.scale_change[0] * self.scale_change[1]))

    def process_after(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        code = ''
        for var_idx, (var, tp) in enumerate(var_types):
            values = []
            for local_idx in range(type_size(tp)):
                idx = 4 * var_idx + local_idx
                prev_x = int((x - center[0]) / self.scale_change[0] + prev_center[0])
                prev_y = int((y - center[1]) / self.scale_change[1] + prev_center[1])
                ch_idx = int(idx * (self.scale_change[0] * self.scale_change[1]) \
                             + ((x - center[0]) % self.scale_change[0]) * self.scale_change[1] \
                             + (y - center[1]) % self.scale_change[1])
                expr, inp_tp = input_exprs[0][prev_x][prev_y][ch_idx // 4]
                values.append(f"({expr}){'.' + 'xyzw'[ch_idx % 4] if type_size(inp_tp) > 1 else ''}")
            code += f"{tp} {var} = {tp}({', '.join(values)});\n"
        return code


class SpaceToDepth(Operation):
    op_type = 'SpaceToDepth'

    def __init__(self, *args, **kwargs):
        super(SpaceToDepth, self).__init__(*args, **kwargs)

        pattern_ops = kwargs.get('pattern_ops')
        shape = self.pattern_ops[0].constants[0].w if pattern_ops else kwargs.get('shape')
        shape = (shape[-3], shape[-1])
        # self.requires_shape = shape
        self.scale_change = tuple(1.0 / dim for dim in shape)

    def _update_channels(self, input_channels):
        assert len(self.input) == 1, \
            f'Get more inputs than 1 for SpaceToDepth operation: {self.name} --- {self.input_names}'
        self.out_channels = int(self.input[0].out_channels / (self.scale_change[0] * self.scale_change[1]))

    def process_after(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        code = ''
        inv_scale = tuple(int(1 / scale_change) for scale_change in self.scale_change)
        for var_idx, (var, tp) in enumerate(var_types):
            values = []
            for local_idx in range(type_size(tp)):
                idx = 4 * var_idx + local_idx
                prev_x = int((x - center[0]) * inv_scale[0] + prev_center[0] \
                             + (idx // inv_scale[0]) % inv_scale[0])
                prev_y = int((y - center[1]) * inv_scale[1] + prev_center[1] \
                             + idx % inv_scale[1])
                ch_idx = int(idx / (inv_scale[0] * inv_scale[1]))
                inp_var, inp_tp = input_exprs[0][prev_x][prev_y][ch_idx // 4]
                values.append(f"{inp_var}{'.' + 'xyzw'[ch_idx % 4] if type_size(inp_tp) > 1 else ''}")
            code += f"{tp} {var} = {tp}({', '.join(values)});\n"
        return code


class Read(Operation):
    op_type = 'read'

    def __init__(self, *args, **kwargs):
        super(Read, self).__init__(*args, **kwargs)

    def _update_channels(self, input_channels):
        if self.name in INPUT_NAMES:
            pass

    def process_before(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        if self.name in INPUT_NAMES:
            texs = INPUT_TEXTURE
        else:
            texs = self.input[0].out_item
        tex_names = texs.get_names()

        for var_idx, (var, tp) in enumerate(var_types):
            step = min(4, self.out_channels - var_idx * 4)
            # calculate reading position
            # (if we running on pass_scale then we need independently load * mul pixels to process all of them)
            muls = tuple(scale / pass_scale for scale, pass_scale in
                         zip(self.scale, get_min_scale(pass_scales[self.depth])))
            pos_mul = f"int2({', '.join(f'pos.{idx}' + (f' * {mul}' if mul != 1 else '') for mul, idx in zip(muls, 'xy'))})"
            pos = f"{pos_mul} + int2({x - center[0]}, {y - center[1]})"

            if self.name in INPUT_NAMES:
                expr = f"load({pos})"
            elif self.read_type == 'resize':
                # if we run on op.scale then it's usual read and we independently load pos * mul pixels but using scale change
                input_size = "float2(input_tex_size.w, input_tex_size.h)" if VULKAN else "GetInputSize()"
                pos = f"float2((({pos}).x + 0.5f) / {self.scale_change[0]}, " \
                      f"(({pos}).y + 0.5f) / {self.scale_change[1]}) / {input_size}"
                expr = f"{tex_names[var_idx]}.SampleLevel({shader_manager.sampler}, {pos}, 0)" \
                       f"{f'.xyzw'[:step + 1] if step < 4 else ''}"
            else:
                expr = f"{tex_names[var_idx]}[{pos}]{f'.xyzw'[:step + 1] if step < 4 else ''}"

            if QUANTIZE_RW and len(self.input) > 0 and self.input[0].quantized:
                quant_values = self.input[0].quantized
                bias, scale = quant_values['bias'], quant_values['scale']
                if QUANTIZE_RW == 'UNORM':
                    expr = f"READ({expr} * {prec(scale, tp)} + {prec(bias, tp)}, {pos}, {self.scale})"
                else:
                    expr = f"({expr} * {prec(scale, tp)})"

            out[x][y][var_idx] = (expr, tp)


class Write(Operation):
    op_type = 'write'

    def __init__(self, *args, **kwargs):
        super(Write, self).__init__(*args, **kwargs)

    def process_after(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        code = ''
        if not self.name.startswith('small_write'):
            if self.name == OUTPUT_WRITE:
                texs = OUTPUT_TEXTURE
            else:
                texs = self.out_item
            input_vec, tex_names = self.input[0].out_item, texs.get_names()

            for inp_idx, (inp_var, inp_tp) in enumerate(input_exprs[0][x][y]):
                step = min(4, input_vec.channels - inp_idx * 4)
                muls = tuple(scale / pass_scale for scale, pass_scale in
                             zip(self.scale, get_min_scale(pass_scales[self.depth])))
                pos = f"int2(pos.x{f' * {muls[0]}' if muls[0] > 1 else ''} + {x - center[0]}, " \
                      f"pos.y{f' * {muls[1]}' if muls[1] > 1 else ''} + {y - center[1]})"
                # output preprocessing before writing
                if self.name == OUTPUT_WRITE:
                    code += f"write({inp_var}, {pos}, float2({self.scale[0]}, {self.scale[1]}));\n"
                else:
                    if step == 4:
                        values = inp_var
                    elif step == 1:
                        values = f"{inp_tp}4({inp_var}, 0, 0, 0)"
                    else:
                        values = f"{inp_tp[:-1]}4("
                        for idx in range(4):
                            if idx < step:
                                values += f"{inp_var}.{'xyzw'[idx]}, "
                            else:
                                values += "0, "
                        values = values[:-2] + ")"  # remove last comma and space

                    if QUANTIZE_RW:
                        stats = ACTIVATIONS[self.input[0].name]
                        mn, mx = stats['01'], stats['99']
                        scale = mx - mn if QUANTIZE_RW == 'UNORM' else max(abs(mn), abs(mx))
                        values = f"{values} - {prec(mn, inp_tp)}" if QUANTIZE_RW == 'UNORM' else values
                        values = f"({values}) / {prec(scale, inp_tp)}"
                        self.quantized = {
                            'bias': mn,
                            'scale': scale
                        }

                    code += f"{tex_names[inp_idx]}[{pos}] = {values};\n"
        return code


class Resize(Read):
    op_type = 'Resize'

    def __init__(self, *args, **kwargs):
        super(Resize, self).__init__(*args, **kwargs)

        weights = kwargs.get('weights')
        self.read_type = 'resize'
        self.sync = True
        if weights and self.input_names[-1] in weights:
            self.scale_change = tuple(float(scale) for scale in weights[self.input_names[-1]][-2:])
            self.input_names = self.input_names[:-1]
        else:
            self.scale_change = kwargs.get('scale_change', (1.0, 1.0))

    def post_init(self):
        self.op_type = 'read'
        if len(self.constants) > 0:
            self.scale_change = tuple(float(scale) for scale in self.constants[0].w[-2:])

    def _update_channels(self, input_channels):
        assert all([ch == input_channels[0] for ch in input_channels]), \
            f'Found different inputs for resize {op.onnx_name} --- {input_channels}'
        self.out_channels = input_channels[0]


class Repeat(Operation):
    op_type = 'Repeat'

    def __init__(self, *args, **kwargs):
        super(Repeat, self).__init__(*args, **kwargs)

        pattern_ops = kwargs.get('pattern_ops')
        self.w = self.pattern_ops[-1].constants[0].w if pattern_ops else kwargs.get('w', (1, 1))
        self.scale_change = tuple(float(dim) for dim in self.w[-2:])

    def _update_channels(self, input_channels):
        assert len(self.input) == 1, \
            f'Get more inputs than 1 for Repeat operation: {self.name} --- {self.input_names}'
        self.out_channels = self.input[0].out_channels * self.w[1]

    def process_before(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        prev_vars = [var for _ in range(self.w[1]) for var in
                     input_exprs[0][
                         int((x - center[0]) / self.w[-2] + prev_center[0])][
                         int((y - center[1]) // self.w[-1] + prev_center[1])]]
        values = concat_variables(prev_vars, out[x][y])
        for var_idx, val in enumerate(values):
            out[x][y][var_idx] = (val, out[x][y][var_idx][1])


class Gather(Operation):
    op_type = 'Gather'

    def __init__(self, *args, **kwargs):
        super(Gather, self).__init__(*args, **kwargs)

        weights = kwargs.get('weights')
        if weights and self.input_names[-1] in weights:
            self.index = weights[self.input_names[-1]]
            self.input_names = self.input_names[:-1]
        else:
            self.index = kwargs.get('index')

    def post_init(self):
        if len(self.constants) > 0:
            self.index = self.constants[0].w

    def _update_channels(self, input_channels):
        assert len(self.input) == 1, \
            f'Get more inputs than 1 for gather operation: {self.name} --- {self.input_names}'
        self.out_channels = 1

    def process_before(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        inp_var, inp_tp = input_exprs[0][x][y][self.index // 4]
        expr = f"{inp_var if type_size(inp_tp) == 1 else f'{inp_var}.' + 'xyzw'[self.index % 4]}"
        out[x][y][0] = (expr, inp_tp)


class Clip(Operation):
    op_type = 'Clip'

    def __init__(self, *args, **kwargs):
        super(Clip, self).__init__(*args, **kwargs)

        weights = kwargs.get('weights')
        if weights and self.input_names[-2] in weights and self.input_names[-1] in weights:
            self.clip_min = weights[self.input_names[-2]]
            self.clip_max = weights[self.input_names[-1]]
            self.input_names = self.input_names[:-2]
        else:
            self.clip_min = kwargs.get('clip_min')
            self.clip_max = kwargs.get('clip_max')

    def post_init(self):
        if len(self.constants) > 0:
            self.clip_min = self.constants[0].w
            self.clip_max = self.constants[1].w


class Slice(Operation):
    op_type = 'Slice'

    def __init__(self, *args, **kwargs):
        super(Slice, self).__init__(*args, **kwargs)

        self.slice_start = kwargs.get('slice_start', 0)
        self.slice_end = kwargs.get('slice_end', 0)
        self.slice_axes = 1
        self.slice_steps = 1

    def post_init(self):
        if len(self.constants) > 0:
            self.slice_start = self.constants[0].w[0]
            self.slice_end = self.constants[1].w[0]
            self.slice_axes = self.constants[2].w[0]
            self.slice_steps = self.constants[3].w[0]
            assert self.slice_axes == 1, \
                f'Slice supported only for channels dimension but got channel {self.slice_axes} for op {self.name}'

    def _update_channels(self, input_channels):
        assert len(self.input) == 1, \
            f'Get more inputs than 1 for slice operation: {self.name} --- {self.input_names}'
        self.out_channels = len(range(self.slice_start, self.slice_end, self.slice_steps))

    def process_before(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        prev_vars = []
        for inp_idx, (inp_var, inp_tp) in enumerate(input_exprs[0][x][y]):
            if 4 * inp_idx + type_size(inp_tp) <= self.slice_start or 4 * inp_idx >= self.slice_end:
                continue
            if 4 * inp_idx >= self.slice_start and 4 * inp_idx + type_size(inp_tp) <= self.slice_end:
                prev_vars.append((inp_var, inp_tp))
            elif 4 * inp_idx < self.slice_start:
                prev_vars.extend([
                    (f"{inp_var}{f'.' + 'xyzw'[idx - 4 * inp_idx] if type_size(inp_tp) > 1 else ''}",
                     inp_tp if type_size(inp_tp) == 1 else inp_tp[:-1])
                    for idx in range(self.slice_start, min(4 * inp_idx + type_size(inp_tp), self.slice_end))])
            elif 4 * inp_idx + type_size(inp_tp) > self.slice_end:
                prev_vars.extend([
                    (f"{inp_var}{f'.' + 'xyzw'[idx - 4 * inp_idx] if type_size(inp_tp) > 1 else ''}",
                     inp_tp if type_size(inp_tp) == 1 else inp_tp[:-1])
                    for idx in range(max(4 * inp_idx, self.slice_start), self.slice_end)])
            else:
                raise NotImplementedError(f"Unexpected slice case {self.name}")
        values = concat_variables(prev_vars, out[x][y])
        for var_idx, val in enumerate(values):
            out[x][y][var_idx] = (val, out[x][y][var_idx][1])


class Reduce(Operation):
    op_type = 'reduce'


class OpChain(Operation):
    op_type = 'OpChain'

    def process_before(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        for var_idx, (var, tp) in enumerate(var_types):
            chain_ops = self.pattern_ops
            step = min(4, self.out_channels - var_idx * 4)
            op_signs = {
                'Add': '+',
                'Mul': '*',
                'Div': '/',
                'Sub': '-'
            }
            if chain_ops[0].op_type in op_signs:
                result = f' {op_signs[chain_ops[0].op_type]} '.join(
                    [var_names[x][y][var_idx][0] for var_names in input_exprs] +
                    [prec(constant.w, tp) for constant in chain_ops[0].constants])
                expr = f"({result})"
            else:
                expr = apply_op(input_exprs[0][x][y][var_idx][0], chain_ops[0], var_idx, tp, step,
                                shader_manager.fun_defines)

            for __op in chain_ops[1:]:
                expr = apply_op(expr, __op, var_idx, tp, step, shader_manager.fun_defines)
            out[x][y][var_idx] = (expr, tp)


class Concat(Operation):
    op_type = 'Concat'

    def _update_channels(self, input_channels):
        self.out_channels = sum([op.out_channels for op in self.input])

    def process_before(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        prev_vars = [var for var_names in input_exprs for var in var_names[x][y]]
        values = concat_variables(prev_vars, out[x][y])
        for var_idx, val in enumerate(values):
            out[x][y][var_idx] = (val, out[x][y][var_idx][1])


class PRelu(Operation):
    op_type = 'PRelu'

    def __init__(self, *args, **kwargs):
        super(PRelu, self).__init__(*args, **kwargs)

        weights = kwargs.get('weights')
        if weights and self.input_names[-1] in weights:
            self.slope = weights[self.input_names[-1]][:, 0, 0]
            self.input_names = self.input_names[:-1]
        else:
            self.slope = kwargs.get('slope')


class LeakyRelu(Operation):
    op_type = 'LeakyRelu'

    def __init__(self, *args, **kwargs):
        super(LeakyRelu, self).__init__(*args, **kwargs)

        onnx_node = kwargs.get('onnx_node')
        self.slope = self.get_attribute(onnx_node, 'alpha').f if onnx_node else kwargs.get('alpha')


class AveragePool(Operation):
    op_type = 'AveragePool'

    def __init__(self, *args, **kwargs):
        super(AveragePool, self).__init__(*args, **kwargs)

        onnx_node = kwargs.get('onnx_node')
        self.strides = tuple(self.get_attribute(onnx_node, "strides").ints) \
            if onnx_node else kwargs.get('strides', (1, 1))
        self.scale_change = tuple(1.0 / stride for stride in self.strides)

    def process_after(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        code = ''
        inv_scale = tuple(int(1 / scale_change) for scale_change in self.scale_change)
        for var_idx, (var, tp) in enumerate(var_types):
            code += f"{tp} {var} = {tp}({f', '.join([f'0' for _ in range(type_size(tp))])});\n"
            for i in range(inv_scale[0]):
                for j in range(inv_scale[1]):
                    prev_var = input_exprs[0][
                        int((x - center[0]) * inv_scale[0] + prev_center[0] + i)][
                        int((y - center[1]) * inv_scale[1] + prev_center[1] + j)][var_idx][0]
                    code += f"{var} += {prev_var};\n"
            code += f"{var} /= {inv_scale[0] * inv_scale[1]};\n"
        return code


class MaxPool(Operation):
    op_type = 'MaxPool'

    def __init__(self, *args, **kwargs):
        kwargs['op_type'] = kwargs.get('op_type', self.op_type)
        super(MaxPool, self).__init__(*args, **kwargs)

        onnx_node = kwargs.get('onnx_node')
        self.strides = tuple(self.get_attribute(onnx_node, "strides").ints) \
            if onnx_node else kwargs.get('strides', (1, 1))
        self.scale_change = tuple(1.0 / stride for stride in self.strides)

    def process_after(self, x, y, var_types, input_exprs, mask, out, prev_center, center, **kwargs):
        code = ''
        inv_scale = tuple(int(1 / scale_change) for scale_change in self.scale_change)
        for var_idx, (var, tp) in enumerate(var_types):
            code += f"{tp} {var} = {tp}({f', '.join([f'-1e9' for _ in range(type_size(tp))])});\n"
            for i in range(inv_scale[0]):
                for j in range(inv_scale[1]):
                    prev_var = input_exprs[0][
                        int((x - center[0]) * inv_scale[0] + prev_center[0] + i)][
                        int((y - center[1]) * inv_scale[1] + prev_center[1] + j)][var_idx][0]
                    code += f"{var} = max({var}, {prev_var});\n"
        return code


class Texture:

    def __init__(self, begin, end, scale=(1.0, 1.0), names=None):
        self.begin = begin
        self.end = end
        self.scale = scale
        self.names = names

        self.required = set()

    def get_tex_name(self, idx, scale):
        scale_key = get_scale_key(scale)
        return f'tex{scale_key}_{idx}'

    def get_names(self):
        if self.names:
            return self.names
        self.names = []

        idx = self.begin
        while idx <= self.end:
            self.names.append(self.get_tex_name(idx // 4, self.scale))
            idx += 4 - (idx % 4)
        return self.names

    def add_requirement(self, depth):
        self.required.add(depth)

    def max_requirement(self):
        return max(self.required, default=-1)


class Shape:

    def __init__(self, dims=2, mask=None):
        self.mask = np.ones(shape=tuple(1 for _ in range(dims)), dtype=bool) \
            if mask is None else Shape.reduce_mask(mask)

    def is_default(self):
        __mask = self.mask.copy()
        center = tuple((dim - 1) // 2 for dim in __mask.shape)
        __mask[center] = False
        return (~__mask).all()

    @staticmethod
    def increase_mask(mask, dim_idx=None):

        if dim_idx is None:
            dim_idxes = list(range(len(mask.shape)))
        else:
            dim_idxes = [dim_idx]

        for dim_idx in dim_idxes:
            pad = np.zeros(tuple(dim if idx != dim_idx else max(dim // 2, 1)
                                 for idx, dim in enumerate(mask.shape)), dtype=bool)
            if mask.shape[dim_idx] == 1:
                mask = np.concatenate([mask, pad], axis=dim_idx)
            else:
                mask = np.concatenate([pad, mask, pad], axis=dim_idx)
        return mask

    @staticmethod
    def equalize_masks(mask1, mask2):
        for dim_idx in range(len(mask1.shape)):
            while mask1.shape[dim_idx] < mask2.shape[dim_idx]:
                mask1 = Shape.increase_mask(mask1, dim_idx)
            while mask2.shape[dim_idx] < mask1.shape[dim_idx]:
                mask2 = Shape.increase_mask(mask2, dim_idx)
        return mask1, mask2

    @staticmethod
    def reduce_mask(mask):
        for dim_idx, _ in enumerate(mask.shape):
            while True:
                if mask.shape[dim_idx] == 1:
                    break
                center = (mask.shape[dim_idx] - 1) // 2
                indices = np.arange(mask.shape[dim_idx])
                # left 2^{b-1} - 1
                left_size = (center + 1) // 2
                # right 2^{b-1} + 1
                right_size = max((mask.shape[dim_idx] - center - 1) // 2, 1)
                if (left_size == 0 or (~np.take(mask, indices[:left_size], axis=dim_idx)).all()) \
                        and (~np.take(mask, indices[-right_size:], axis=dim_idx)).all():
                    mask = np.take(mask, indices[left_size:-right_size], axis=dim_idx)
                else:
                    break
        return mask

    @staticmethod
    def fix_mask_dim(mask, center, dim_idx):

        # left 2^{b-1}-1
        # right 2^{b-1}+1
        dim = mask.shape[dim_idx]
        left_b = len(bin(center)[2:]) if center > 0 else 0
        right_b = len(bin(dim - center - 2)[2:]) if dim - center - 2 > 0 else 0
        bits = max(left_b, right_b) + 1

        left_pad = 2 ** (bits - 1) - 1 - (center)
        if left_pad > 0:
            mask = np.concatenate([np.zeros(tuple(dim if idx != dim_idx else left_pad
                                                  for idx, dim in enumerate(mask.shape)), dtype=bool), mask],
                                  axis=dim_idx)
        right_pad = 2 ** (bits - 1) + 1 - (dim - center)
        if right_pad > 0:
            mask = np.concatenate([mask, np.zeros(tuple(dim if idx != dim_idx else right_pad
                                                        for idx, dim in enumerate(mask.shape)), dtype=bool)],
                                  axis=dim_idx)

        return mask

    @staticmethod
    def create_mask(shape, mask_type='conv'):
        mask = np.ones(shape, dtype=bool)
        if all(dim == 1 for dim in shape):
            return Shape(mask=mask)

        for dim_idx, dim in enumerate(shape):
            center = (dim - 1) // 2 if mask_type == 'conv' else 0
            mask = Shape.fix_mask_dim(mask, center, dim_idx)

        return Shape(mask=mask)

    def __or__(self, shape):
        mask1, mask2 = Shape.equalize_masks(self.mask, shape.mask)
        return Shape(mask=mask1 | mask2)

    @staticmethod
    def mask_pad(mask, *pads, dilations=None):

        if dilations is None:
            dilations = tuple(1 for _ in range(len(mask.shape)))

        # increase dimensions to add paddings
        for dim_idx, _ in enumerate(mask.shape):
            filled_axes = mask.sum(axis=tuple(idx for idx, dim in enumerate(mask.shape) if idx != dim_idx), dtype=bool)
            while np.argmax(filled_axes) < pads[2 * dim_idx] * dilations[dim_idx] or \
                    np.argmax(np.flip(filled_axes)) < pads[2 * dim_idx + 1] * dilations[dim_idx]:
                mask = Shape.increase_mask(mask, dim_idx)
                filled_axes = mask.sum(axis=tuple(idx for idx, dim in enumerate(mask.shape) if idx != dim_idx),
                                       dtype=bool)
        # add padding from each side
        result_mask = mask.copy()
        for dim_idx, dim in enumerate(mask.shape):
            indices = np.arange(dim)
            for pad_idx in range(pads[dim_idx * 2]):
                step = (pad_idx + 1) * dilations[dim_idx]
                result_mask |= np.concatenate([
                    np.take(mask, indices=indices[step:], axis=dim_idx),
                    np.zeros(tuple(d if idx != dim_idx else step for idx, d in enumerate(mask.shape)), dtype=bool)],
                    axis=dim_idx)
            for pad_idx in range(pads[dim_idx * 2 + 1]):
                step = (pad_idx + 1) * dilations[dim_idx]
                result_mask |= np.concatenate([
                    np.zeros(tuple(d if idx != dim_idx else step for idx, d in enumerate(mask.shape)), dtype=bool),
                    np.take(mask, indices=indices[:-step], axis=dim_idx)], axis=dim_idx)
            mask = result_mask.copy()

        return result_mask

    def kernel_pad(self, ks, dilations=None):
        return Shape(mask=Shape.mask_pad(self.mask, *[pad for dim in ks for pad in [(dim - 1) // 2, dim // 2]],
                                         dilations=dilations))

    @staticmethod
    def scale_mask(mask, scale_change):
        centers, pads = [], []
        for dim_idx, dim in enumerate(mask.shape):
            center = (mask.shape[dim_idx] - 1) // 2
            mask = np.take(mask, indices=np.arange(
                start=center % scale_change[dim_idx], stop=dim - (dim - center) % scale_change[dim_idx]), axis=dim_idx)
            centers.append(center // scale_change[dim_idx])
            pads.extend([center % scale_change[dim_idx], (dim - center) % scale_change[dim_idx]])

        cmask = convolve(mask, np.ones(tuple(scale_change[idx] for idx in range(len(mask.shape)))), mode='valid') > 0

        for dim_idx, dim in enumerate(cmask.shape):
            cmask = np.concatenate([
                cmask, np.zeros(tuple(d if idx != dim_idx else mask.shape[dim_idx] - dim
                                      for idx, d in enumerate(cmask.shape)), dtype=bool)], axis=dim_idx)
            cmask = np.take(cmask, indices=np.arange(start=0, stop=cmask.shape[dim_idx], step=scale_change[dim_idx]),
                            axis=dim_idx)
        return cmask, centers, pads

    def scale_pad(self, scale_change):

        if all(scale >= 1.0 for scale in scale_change):
            return deepcopy(self)

        invert_scales = tuple(int(1 / scale) for scale in scale_change)
        mask = self.mask.copy()

        # increase dimensions to add paddings
        for dim_idx, _ in enumerate(mask.shape):
            filled_axes = mask.sum(axis=tuple(idx for idx, dim in enumerate(mask.shape) if idx != dim_idx), dtype=bool)
            center = (mask.shape[dim_idx] - 1) // 2
            while np.argmax(filled_axes) < center % invert_scales[dim_idx] or \
                    np.argmax(np.flip(filled_axes)) < (mask.shape[dim_idx] - center) % invert_scales[dim_idx]:
                mask = Shape.increase_mask(mask, dim_idx)
                filled_axes = mask.sum(axis=tuple(idx for idx, dim in enumerate(mask.shape) if idx != dim_idx),
                                       dtype=bool)
                center = (mask.shape[dim_idx] - 1) // 2

        # cut pads and scale down
        cmask, centers, pads = Shape.scale_mask(mask, invert_scales)

        # scale up and add removed pads
        for dim_idx, scale in enumerate(invert_scales):
            cmask = np.swapaxes(cmask, axis1=0, axis2=dim_idx)
            cmask = np.repeat(cmask[:, np.newaxis], scale, axis=1).reshape((-1,) + cmask.shape[1:])
            cmask = np.swapaxes(cmask, axis1=dim_idx, axis2=0)
            if pads[dim_idx * 2] > 0:
                cmask = np.concatenate([
                    np.zeros(tuple(d if idx != dim_idx else pads[dim_idx * 2]
                                   for idx, d in enumerate(cmask.shape)), dtype=bool), cmask], axis=dim_idx)
            if pads[dim_idx * 2 + 1] > 0:
                cmask = np.concatenate([cmask,
                                        np.zeros(tuple(d if idx != dim_idx else pads[dim_idx * 2 + 1] for idx, d in
                                                       enumerate(cmask.shape)), dtype=bool)], axis=dim_idx)

        return Shape(mask=cmask)

    def scale(self, scale_change):

        mask = self.scale_pad(scale_change).mask

        if all(scale <= 1.0 for scale in scale_change):
            mask, centers, _ = Shape.scale_mask(mask, tuple(int(1 / scale) for scale in scale_change))
            for dim_idx, dim in enumerate(mask.shape):
                mask = Shape.fix_mask_dim(mask, centers[dim_idx], dim_idx)
        else:
            for dim_idx, scale in enumerate(scale_change):
                center = (mask.shape[dim_idx] - 1) // 2
                mask = np.swapaxes(mask, axis1=0, axis2=dim_idx)
                mask = np.repeat(mask[:, np.newaxis], scale, axis=1).reshape((-1,) + mask.shape[1:])
                mask = np.swapaxes(mask, axis1=dim_idx, axis2=0)
                mask = Shape.fix_mask_dim(mask, int(center * scale), dim_idx)

        return Shape(mask=mask)

    def __eq__(self, other):
        if not isinstance(other, Shape):
            return False
        mask1, mask2 = Shape.equalize_masks(self.mask, other.mask)
        return np.array_equal(mask1, mask2)

    def __hash__(self):
        return hash(self.mask.data.tobytes())


class Vector:

    def __init__(self, channels, op_idx, shape, base_type='float'):
        self.shape = shape
        self.op_idx = op_idx
        self.channels = channels
        self.base_type = base_type

        self.exprs = None

    def create_exprs(self):
        mask = self.shape.mask
        exprs = np.empty(mask.shape, dtype=list)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x][y]:
                    pos_x = x - (mask.shape[0] - 1) // 2
                    pos_y = y - (mask.shape[1] - 1) // 2
                    exprs[x][y] = [(f'var{self.op_idx}_'
                                    f'{pos_x if pos_x >= 0 else f"m{abs(pos_x)}"}_'
                                    f'{pos_y if pos_y >= 0 else f"m{abs(pos_y)}"}_'
                                    f'{step}',
                                    f'{self.base_type}{"" if self.channels - 4 * step == 1 else min(4, self.channels - 4 * step)}')
                                   for step in range(self.channels // 4 + int(self.channels % 4 != 0))]
        return exprs

    def get_exprs(self):
        if self.exprs is None:
            self.exprs = self.create_exprs()
        return self.exprs

    def set_exprs(self, exprs):
        self.exprs = exprs


weights = {item.name: np.frombuffer(item.raw_data, dtype=dtype_mapper(item.data_type)).reshape(item.dims) for item in
           list(onnx_graph.initializer)}

shader_manager = ShaderManager(header=HEADER_FILE, block_size=BLOCK_SIZE)
OP_CLASSES = defaultdict(lambda: Operation, {cls.op_type: cls for cls in all_subclasses(Operation)})
ops = {op.name: op for op in [(OP_CLASSES[nd.op_type])(
    op_type=nd.op_type, name=nd.output[0], onnx_node=nd, input_names=list(nd.input), weights=weights)
    for nd in onnx_graph.node]}

INPUT_NAMES = []
input_ops = [op for op in ops.values() if INPUT_NAME in op.input_names]
for op in input_ops:
    name = f'{INPUT_NAME}_{op.name}'
    ops[name] = Read(name=name)
    ops[name].out_channels = int(onnx_graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    for idx in [idx for idx, item in enumerate(op.input_names) if item == INPUT_NAME]:
        op.input_names[idx] = name
    INPUT_NAMES.append(name)

OUTPUT_WRITE = f'{OUTPUT_NAME}_write'
ops[OUTPUT_WRITE] = Write(name=OUTPUT_WRITE, input_names=(OUTPUT_NAME,))
for op in ops.values():

    # initialize connections
    for name in op.input_names:
        inp = ops[name]
        # merge all constants to parents
        if inp.op_type == 'Constant':
            op.constants.append(inp)
        else:
            op.input.append(ops[name])
            inp.output.append(op)
# filter extra ops from graph
ops = {op.name: op for op in ops.values() if op.op_type not in ('Constant',)}
for op in ops.values():
    op.post_init()


# find patterns in graph
class PatternState(Enum):
    INITIAL = 1
    OP_CHAIN_START = 2
    OP_CHAIN = 3
    SPACE2DEPTH_RESHAPE1 = 4
    SPACE2DEPTH_TRANSPOSE = 5
    REPEAT_EXPAND = 6
    SILU = 7
    GELU_DIV = 8
    GELU_ERF = 9
    GELU_ADD = 10
    GELU_MUL = 11


def in_opchain(name):
    return name.lower() in ('mul', 'div', 'add', 'sub', 'clip', 'leakyrelu', 'prelu') \
           + tuple(shader_manager.fun_defines)


def find_patterns(op, state, visited, pattern=None):
    if state == PatternState.INITIAL:
        visited.add(op)

    patterns, state_patterns = defaultdict(list), []
    for nxt in op.output:
        if state == PatternState.INITIAL:
            # chain of operations
            if in_opchain(nxt.op_type):
                if len(nxt.output) == 1:
                    patterns['OpChain'].extend(find_patterns(nxt, PatternState.OP_CHAIN, visited, pattern=[nxt]))
                else:
                    patterns['OpChain'].append([nxt])
            # silu op
            if nxt.op_type == 'Sigmoid':
                patterns['Silu'].extend(find_patterns(nxt, PatternState.SILU, visited, pattern=[nxt]))
            # SpaceToDepth
            if nxt.op_type == 'Reshape':
                patterns['SpaceToDepth'].extend(
                    find_patterns(nxt, PatternState.SPACE2DEPTH_RESHAPE1, visited, pattern=[nxt]))
            # GELU
            if nxt.op_type == 'Div' and len(nxt.constants) and (nxt.constants[0].w - 1.41) < 1e-2:
                patterns['Gelu'].extend(find_patterns(nxt, PatternState.GELU_DIV, visited, pattern=[nxt]))

            # Repeat channels
            if nxt.op_type == 'Expand':
                patterns['Repeat'].extend(
                    find_patterns(nxt, PatternState.REPEAT_EXPAND, visited, pattern=[nxt]))
            # try all other operations if they could be a start of pattern
            if nxt not in visited:
                patterns = merge_list_dicts(patterns, find_patterns(nxt, PatternState.INITIAL, visited))
        elif state == PatternState.OP_CHAIN:
            if in_opchain(nxt.op_type):
                if len(nxt.input) == 1 and len(nxt.output) == 1:
                    state_patterns.extend(find_patterns(nxt, PatternState.OP_CHAIN, visited, pattern=pattern + [nxt]))
                elif len(nxt.input) == 1:
                    state_patterns.append(pattern + [nxt])
        elif state == PatternState.SILU:
            if nxt.op_type == 'Mul' and set(nxt.input) == {pattern[0], pattern[0].input[0]}:
                state_patterns.append(pattern + [nxt])
        elif state == PatternState.GELU_DIV:
            if nxt.op_type == 'Erf':
                state_patterns.extend(find_patterns(nxt, PatternState.GELU_ERF, visited, pattern=pattern + [nxt]))
        elif state == PatternState.GELU_ERF:
            if nxt.op_type == 'Add' and len(nxt.constants) and nxt.constants[0].w == 1:
                state_patterns.extend(find_patterns(nxt, PatternState.GELU_ADD, visited, pattern=pattern + [nxt]))
        elif state == PatternState.GELU_ADD:
            if nxt.op_type == 'Mul' and set(nxt.input) == {pattern[-1], pattern[0].input[0]}:
                state_patterns.extend(find_patterns(nxt, PatternState.GELU_MUL, visited, pattern=pattern + [nxt]))
        elif state == PatternState.GELU_MUL:
            if nxt.op_type == 'Mul' and len(nxt.constants) and nxt.constants[0].w == 0.5:
                state_patterns.append(pattern + [nxt])
        elif state == PatternState.SPACE2DEPTH_RESHAPE1:
            if nxt.op_type == 'Transpose':
                state_patterns.extend(
                    find_patterns(nxt, PatternState.SPACE2DEPTH_TRANSPOSE, visited, pattern=pattern + [nxt]))
        elif state == PatternState.SPACE2DEPTH_TRANSPOSE:
            if nxt.op_type == 'Reshape':
                state_patterns.append(pattern + [nxt])
        elif state == PatternState.REPEAT_EXPAND:
            if nxt.op_type == 'Tile':
                state_patterns.append(pattern + [nxt])

    if state == PatternState.INITIAL:
        return patterns

    # if there is no next op chains then we found the maximum one
    if state == PatternState.OP_CHAIN and len(state_patterns) == 0:
        state_patterns.append(pattern)

    return state_patterns


last_patterns = []
for pattern_iter in range(len(ops)):
    visited, pattern_used_ops = set(), set()
    patterns = merge_list_dicts(
        *[find_patterns(ops[name], PatternState.INITIAL, visited=visited) for name in INPUT_NAMES])
    patterns = [(name, __pattern) for name, patterns_lst in patterns.items()
                for __pattern in patterns_lst if len(__pattern) > 0]
    patterns = sorted(patterns, key=lambda pair: (len(pair[1]), pair[1][-1].name), reverse=True)

    if pattern_iter > 0 and len(last_patterns) == len(patterns) and \
            all([tuple(p1[1]) == tuple(p2[1]) for p1, p2 in zip(last_patterns, patterns)]):
        break

    for name, __pattern in patterns:
        if any([item in pattern_used_ops for item in __pattern]):
            continue
        pattern_used_ops.update(__pattern)

        for op in __pattern:
            ops.pop(op.name)

        op = OP_CLASSES[name](name=__pattern[-1].name, op_type=name, pattern_ops=__pattern)

        for el in __pattern:
            for __op in el.input:
                if __op not in __pattern:
                    for idx in [idx for idx, item in enumerate(__op.output) if item == el]:
                        __op.output[idx] = op
                    __op.output = [*dict.fromkeys(__op.output)]
                    op.input.append(__op)

            for __op in el.output:
                if __op not in __pattern:
                    for idx in [idx for idx, item in enumerate(__op.input) if item == el]:
                        __op.input[idx] = op
                    op.output.append(__op)

        op.input = [*dict.fromkeys(op.input)]
        op.output = [*dict.fromkeys(op.output)]
        ops[op.name] = op

    last_patterns = patterns

# decompose some blocks
for op in list(ops.values()):
    # split grouped convolution to independent convolutions
    if op.op_type in ('Conv', 'ConvTranspose') and \
            op.groups != 1 and (op.groups != op.w.shape[0] or op.w.shape[1] > 1):
        op.input[0].output = []
        inp_size = op.w.shape[1] if op.op_type == 'Conv' else op.w.shape[0] // op.groups
        out_size = op.w.shape[0] // op.groups if op.op_type == 'Conv' else op.w.shape[1]
        convs = []
        for group_id in range(op.groups):
            # take channels for local conv
            slc = Slice(name=f'{op.name}_{group_id}_slice', inp=op.input,
                        out_channels=inp_size, slice_start=group_id * inp_size, slice_end=(group_id + 1) * inp_size)
            ops[slc.name] = slc
            op.input[0].output.append(slc)
            # take weights for local conv
            w = op.w[group_id * out_size:(group_id + 1) * out_size]
            b = op.b[group_id * out_size: (group_id + 1) * out_size]
            conv = Conv(name=f'{op.name}_{group_id}_conv', inp=[slc],
                        out_channels=out_size, w=w, b=b, groups=1, strides=op.strides)
            ops[conv.name] = conv
            slc.output = [conv]
            convs.append(conv)
        # concat results of local convs
        concat = Concat(name=f'{op.name}_concat', inp=convs, output=op.output,
                        out_channels=op.out_channels)
        ops[concat.name] = concat

        for conv in convs:
            conv.output = [concat]

        for out in op.output:
            for idx in [idx for idx, item in enumerate(out.input) if item == op]:
                out.input[idx] = concat
        ops.pop(op.name)


def backward_dfs(op, __ops, visited):
    visited.add(op)
    for inp in op.input:
        if inp not in visited:
            backward_dfs(inp, __ops, visited)
    __ops.append(op)


def get_backward_order(ops):
    __ops = []
    backward_dfs(ops[OUTPUT_WRITE], __ops, visited=set())
    return list(reversed(__ops))


def forward_dfs(op, __ops, visited):
    visited.add(op)
    for nxt in op.output:
        if nxt not in visited:
            forward_dfs(nxt, __ops, visited)
    __ops.append(op)


def get_forward_order(ops):
    __ops = []
    visited = set()
    for input_name in INPUT_NAMES:
        forward_dfs(ops[input_name], __ops, visited=visited)
    return list(reversed(__ops))


# pass order
pass_scales = defaultdict(list)
in_channels, out_channels = defaultdict(lambda: 0), defaultdict(lambda: 0)
for op in reversed(get_backward_order(ops)):

    prev_scales = [inp.scale for inp in op.input]
    assert all([scale == prev_scales[0] for scale in prev_scales]), \
        f'All scales provided to {op.name} must match but found {prev_scales}'
    op.scale = tuple(float(scale * scale_change) for scale, scale_change in
                     zip(prev_scales[0] if len(prev_scales) > 0 else op.scale, op.scale_change))

    op.depth = max([__op.depth for __op in op.input], default=op.depth)

    if op.requires_synchronization():
        op.update_depth(op.depth + 1, pass_scales)
    else:
        pass_scales[op.depth].append(op.get_run_scale())

    op.update_channels()

if len([op for op in ops.values() if
        (op.op_type == 'read' and op.depth == 1) or
        (op.requires_synchronization() and op.depth == 0)]) == 0:
    # merge first depth without sync operations to first pass
    for op in reversed(get_backward_order(ops)):
        if op.depth > 0:
            op.update_depth(op.depth - 1, pass_scales)

# trying to merge back items to pass
if MERGE_BACK:
    for op in get_backward_order(ops):
        for inp in [item for item in op.input if item.depth < op.depth and not item.requires_synchronization()
                                                 and not any(isinstance(nxt, Read) for nxt in item.output)]:
            if all([nxt.depth == op.depth for nxt in inp.output]):
                inp.update_depth(op.depth, pass_scales)


def fix_depths(ops, depths):
    pass


GREEDY_FORWARD = False
if GREEDY_FORWARD:
    for op in get_forward_order(ops):
        while True:
            depths = {op: op.depth for op in ops.values()}
            depths[op] = min([__op.depth for __op in op.output], default=depths[op])
            if depths[op] == op.depth:
                break
            fix_depths(ops, depths)
            old_score, new_score = 0, 0
            for __op in depths.keys():
                for inp in op.input:
                    if inp.depth != __op.depth:
                        old_score += round4(inp.out_channels) // 4
                    if depths[inp] != depths[__op]:
                        new_score += round4(inp.out_channels) // 4

            if old_score > new_score:
                for __op, depth in depths.items():
                    __op.update_depth(depth, pass_scales)
            else:
                break

# merge input names in same pass to the one input
__INPUT_NAMES = []
for depth, names in groupby(sorted(INPUT_NAMES, key=lambda name: ops[name].depth), key=lambda name: ops[name].depth):
    names = list(names)
    outputs = [*dict.fromkeys([op for name in names for op in ops[name].output])]
    read_sample = ops[names[0]]
    read = Read(name=f'{INPUT_NAME}_{depth}_read',
                output=outputs, depth=depth, scale=read_sample.scale, out_channels=read_sample.out_channels)
    ops[read.name] = read

    for out in outputs:
        for idx in [idx for idx, op in enumerate(out.input) if op.name in names]:
            out.input[idx] = read

    for name in names:
        ops.pop(name)

    __INPUT_NAMES.append(read.name)

INPUT_NAMES = __INPUT_NAMES

# add read/write ops between connected ops in different passes
for op in list(ops.values()):
    connections = defaultdict(list)
    for nxt in op.output:
        if op.depth != nxt.depth:
            connections[nxt.depth].append(nxt)
    if len(connections) == 0:
        continue

    write = Write(name=f'{op.name}_write', input_names=(op.name,), inp=[op],
                  depth=op.depth, scale=op.scale, out_channels=op.out_channels)
    ops[write.name] = write

    for depth, nxt_ops in connections.items():

        for nxt in nxt_ops:
            op.output[op.output.index(nxt)] = write

        nxt_read = [__op for __op in nxt_ops if __op.op_type == 'read']
        for read in nxt_read:
            for idx in [idx for idx, item in enumerate(read.input) if item == op]:
                read.input[idx] = write
            write.output.append(read)

        nxt_ops = [__op for __op in nxt_ops if __op.op_type != 'read']
        if len(nxt_ops) > 0:
            read = Read(name=f'{op.name}_{depth}_read', input_names=(write.name,), inp=[write],
                        output=nxt_ops, depth=depth, scale=op.scale, out_channels=op.out_channels)
            ops[read.name] = read

            for nxt in nxt_ops:
                for idx in [idx for idx, item in enumerate(nxt.input) if item == op]:
                    nxt.input[idx] = read

            # connect write and read
            write.output.append(read)


# merge small read/writes
def merge_group(write_group):
    base_name = '_'.join([op.name for op in write_group])
    depth, scale = write_group[0].depth, write_group[0].scale
    out_channels = sum([op.out_channels for op in write_group])

    # concat all previous writes
    concat = Concat(name=f'{base_name}_concat', inp=[op.input[0] for op in write_group],
                    out_channels=out_channels, depth=depth, scale=scale)
    ops[concat.name] = concat
    for inp, write_op in zip(concat.input, write_group):
        for idx in [idx for idx, op in enumerate(inp.output) if op == write_op]:
            inp.output[idx] = concat
        ops.pop(write_op.name)

    # write one vector
    write = Write(name=f'{base_name}_write', inp=[concat],
                  depth=depth, scale=scale, out_channels=out_channels)
    ops[write.name] = write
    concat.output = [write]

    # calculate actual channels for each read operation
    read_ops = {}
    offset = 0
    for write_op in write_group:
        for read_op in write_op.output:
            read_ops[read_op] = (offset, offset + write_op.out_channels)
        offset += write_op.out_channels

    for depth, read_group in groupby(read_ops.keys(), key=lambda op: op.depth):
        # read one vector for each depth
        read = Read(name=f'{base_name}_{depth}_read', inp=[write],
                    depth=depth, scale=scale, out_channels=out_channels)
        ops[read.name] = read
        write.output.append(read)

        for read_id, read_op in enumerate(read_group):
            segment = read_ops[read_op]
            slc = Slice(name=f'{base_name}_{read_id}_slice', inp=[read], scale=scale, depth=depth,
                        output=read_op.output, out_channels=segment[1] - segment[0],
                        slice_start=segment[0], slice_end=segment[1])
            ops[slc.name] = slc
            read.output.append(slc)

            for out in read_op.output:
                for idx in [idx for idx, op in enumerate(out.input) if op == read_op]:
                    out.input[idx] = slc

            ops.pop(read_op.name)


for (depth, scale), write_ops in groupby(
        [op for op in ops.values() if op.op_type == 'write' and op.name != OUTPUT_WRITE],
        key=lambda op: (op.depth, op.scale)):
    write_ops = sorted(write_ops, key=lambda op: (op.out_channels,
                                                  tuple(sorted([read_op.depth for read_op in op.output]))))
    write_group = []
    for write_op in write_ops:
        # if we could add new op just add and continue
        if sum([op.out_channels for op in write_group]) + write_op.out_channels <= 4:
            write_group.append(write_op)
            continue
        # if we can't add then merge current group
        if len(write_group) > 1:
            merge_group(write_group)

        write_group.clear()
        write_group.append(write_op)

    if len(write_group) > 1:
        merge_group(write_group)

# operations order
for idx, op in enumerate(reversed(get_backward_order(ops))):
    op.priority = idx * 10

# calculate min output scale for each pass
min_scales = defaultdict(lambda: (1e6, 1e6))
for op in reversed(get_backward_order(ops)):
    if op.op_type == 'write':
        min_scales[op.depth] = tuple(min(scale, curr_scale)
                                     for scale, curr_scale in zip(op.scale, min_scales[op.depth]))

depths = list(sorted(min_scales.keys()))

# add extra texture for regulating run size
if MAGPIE:
    for depth in depths:

        if any(pass_scale < write_scale
               for pass_scale, write_scale in zip(get_min_scale(pass_scales[depth]), min_scales[depth])):
            # last op is always write and we add extra output for previous node of this write
            last_op = sorted([op for op in ops.values() if op.depth == depth], key=lambda op: op.priority)[-1].input[0]
            small_write = Write(name=f'small_write_{depth}', depth=last_op.depth, out_channels=1,
                                priority=allocate_after(ops, last_op), scale=get_min_scale(pass_scales[depth]),
                                inp=[last_op])
            ops[small_write.name] = small_write
            last_op.output.append(small_write)

            # for last pass we also must add extra read/write pass in this case
            if depth == depths[-1]:
                out_write, out_op = ops[OUTPUT_WRITE], ops[OUTPUT_WRITE].input[0]
                out_write.update_depth(out_write.depth + 1, pass_scales)
                extra_read = Read(name=f'extra_read_{out_write.depth}', depth=out_write.depth,
                                  out_channels=out_write.out_channels, priority=allocate_before(ops, out_write),
                                  scale=out_write.scale, output=[out_write])
                ops[extra_read.name] = extra_read
                add_between(extra_read, out_op, out_write)

                extra_write = Write(name=f'extra_write_{depth}', depth=out_op.depth,
                                    out_channels=out_op.out_channels, priority=allocate_after(ops, out_op),
                                    scale=out_op.scale,
                                    inp=[out_op], output=[extra_read])
                ops[extra_write.name] = extra_write
                add_between(extra_write, out_op, extra_read)

                min_scales[depth + 1] = min_scales[depth]

            min_scales[depth] = get_min_scale(pass_scales[depth])

# backward propagation for requires_shape
for op in get_backward_order(ops):

    # write operations always have shape determined by run scale
    if op.op_type == 'write':
        op.read_shape = Shape.create_mask(tuple(int(scale / pass_scale) for scale, pass_scale
                                                in zip(op.scale, get_min_scale(pass_scales[op.depth]))), 'scale')
    elif op.requires_shape > (1, 1):
        op.read_shape = op.out_item.shape.kernel_pad(op.requires_shape, op.dilations)
    elif op.op_type != 'read':
        # fix dimension if it's not divisible by scale_change
        op.read_shape = op.out_item.shape.scale(tuple(1 / scale_change for scale_change in op.scale_change))
        op.out_item.shape = op.read_shape.scale(op.scale_change)
    else:
        op.read_shape = op.out_item.shape

    if op.op_type != 'read':
        for inp in op.input:

            # setting maximum required shape for read
            if inp.out_item is None:
                inp.out_item = Vector(inp.out_channels, op_idx=inp.priority, shape=Shape.create_mask((1, 1)),
                                      base_type=BASE_TYPE)

            inp.out_item.shape |= op.read_shape

    # as we have one read per pass for same output then it may have different shape outputs
    next_shapes = set([nxt.read_shape for nxt in op.output])
    if op.op_type != 'write' and len(next_shapes) >= 1:
        for nxt in [nxt for nxt in op.output if nxt.read_shape != op.out_item.shape]:
            priority = allocate_before(ops, nxt)
            reduce = Reduce(name=f'{op.name}_{nxt.name}_reduce', priority=priority, read_shape=op.out_item.shape,
                            scale=op.scale, depth=nxt.depth, out_channels=op.out_channels,
                            out_item=Vector(op.out_channels, op_idx=priority, shape=nxt.read_shape,
                                            base_type=BASE_TYPE))
            ops[reduce.name] = reduce
            add_between(reduce, op, nxt)


# first forward for textures
def texture_dfs(op, used, texture_manager):
    used.add(op)

    if op.op_type == 'write' and op.name != f'{OUTPUT_NAME}_write':
        tex = texture_manager.get_textures(channels=op.out_channels, depth=op.depth, scale=op.scale)
        op.out_item = tex
        for nxt in op.output:
            tex.add_requirement(nxt.depth)

    for nxt in op.output:
        if nxt not in used and nxt.depth == op.depth and all([prev in used for prev in nxt.input]):
            texture_dfs(nxt, used, texture_manager)


def depth_order(ops):
    return groupby(sorted(ops.values(), key=lambda op: op.depth), key=lambda op: op.depth)


def pass_order(pass_ops):
    return sorted(pass_ops, key=lambda op: op.priority)


texture_manager = TextureManager()
used = set()
for depth, pass_ops in depth_order(ops):
    for op in pass_order(pass_ops):
        if op not in used:
            texture_dfs(op, used, texture_manager)


def process_op(op, shader_manager):
    shader_manager.add_pass_text(op.process(), op.depth)


# forward for write operations in correct order
def op_dfs(op, used, shader_manager):
    used.add(op)

    # add textures to passes
    if op.op_type == 'write' and op.name != f'{OUTPUT_NAME}_write':
        tex = op.out_item
        shader_manager.add_write_tex(tex, op.depth)
        for nxt in op.output:
            shader_manager.add_read_tex(tex, nxt.depth)

    process_op(op, shader_manager)

    for nxt in op.output:
        if nxt not in used and nxt.depth == op.depth and all([prev in used for prev in nxt.input]):
            op_dfs(nxt, used, shader_manager)


INPUT_TEXTURE = Texture(begin=0, end=ops[INPUT_NAMES[0]].out_channels - 1, scale=1, names=['INPUT'])
shader_manager.add_read_tex(INPUT_TEXTURE, depth=0)
shader_manager.add_read_tex(INPUT_TEXTURE, depth=ops[OUTPUT_WRITE].depth)
OUTPUT_TEXTURE = Texture(begin=0, end=ops[OUTPUT_WRITE].out_channels - 1, scale=ops[OUTPUT_WRITE].scale,
                         names=['OUTPUT'])
shader_manager.add_write_tex(OUTPUT_TEXTURE, depth=ops[OUTPUT_WRITE].depth)

used = set()
for depth, pass_ops in depth_order(ops):
    for op in pass_order(pass_ops):
        if op not in used:
            op_dfs(op, used, shader_manager)

shader = shader_manager.compile()
with open(OUTPUT_FILE, 'w') as f:
    f.write(shader)
# print(shader)

# TODO improve forward merge euristics
# TODO solve texture allocating problem optimally with known query history
