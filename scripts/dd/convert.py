import os
import textwrap
from collections import defaultdict
from enum import Enum
from itertools import groupby

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import onnx

MODEL_ONNX = '../bad_nn.onnx'
OUTPUT_FILE = 'D:\\PycharmProjects\\explore\\MAGPIE\\effects\\reduce_full.hlsl'
model = onnx.load(MODEL_ONNX)
onnx.checker.check_model(model)
onnx_graph = model.graph
with open('reduce.graph', 'w') as f:
    f.write(str(onnx_graph))

INPUT_NAME = 'input'
OUTPUT_NAME = 'output'


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
        self.__add_text(self.header.split('//!COMMON')[0])

    def prepare_common(self):
        self.__add_text("//!COMMON")
        self.__add_text(self.header.split('//!COMMON')[1].split('//!BEGIN')[0])

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
            //!WIDTH INPUT_WIDTH {f'* {self.scales[name]}' if self.scales[name] != 1 else ''}
            //!HEIGHT INPUT_HEIGHT {f'* {self.scales[name]}' if self.scales[name] != 1 else ''}
            //!FORMAT R16G16B16A16_FLOAT
            Texture2D {name};
            """))

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
        if pass_id == min(self.passes.keys()):
            pass_header += self.header.split('//!BEGIN')[1]
        elif pass_id == max(self.passes.keys()):
            pass_header += self.header.split('//!END')[1]
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
                line = line.replace('INPUT_WIDTH', f'INPUT_WIDTH{f" * {output_scale}" if output_scale != 1 else ""}')
            if '//!OUTPUT_HEIGHT' in line:
                output_scale = self.scales['OUTPUT']
                line = line.replace('INPUT_HEIGHT', f'INPUT_HEIGHT{f" * {output_scale}" if output_scale != 1 else ""}')
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


def get_attribute(onnx_node, name):
    return [attr for attr in onnx_node.attribute if attr.name == name][0]


def round_4(val):
    return val if val % 4 == 0 else val + 4 - (val % 4)


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


weights = {item.name: np.frombuffer(item.raw_data, dtype=dtype_mapper(item.data_type)).reshape(item.dims) for item in
           list(onnx_graph.initializer)}


class Operation:

    def __init__(self, op_type, name, input_names=(), weights=None, depth=0, priority=0, inp=None, scale=1,
                 output=None, out_channels=None, out_item=None, onnx_node=None, scale_change=1, constants=None,
                 pattern_ops=None, sync=False, requires_shape=(1, 1), read_type=None, root_op=None):
        self.op_type = op_type
        self.name = name
        self.input_names = [name for name in input_names if len(name) > 0]
        self.input = inp if inp else list()
        self.constants = constants if constants else list()
        self.output = output if output else list()

        self.depth = depth
        self.priority = priority
        self.scale = scale
        self.sync = sync
        self.requires_shape = requires_shape
        self.read_type = read_type
        self.scale_change = scale_change
        self.root_op = self if op_type == 'read' else root_op

        self.out_channels = out_channels
        self.out_item = out_item if out_item else None

        self.pattern_ops = pattern_ops

        if self.op_type in ('Conv', 'ConvTranspose'):
            self.w = weights[self.input_names[-2]]
            self.b = weights[self.input_names[-1]]
            self.groups = get_attribute(onnx_node, "group").i
            self.strides = tuple(get_attribute(onnx_node, "strides").ints)
            self.input_names = self.input_names[:-2]
            if self.op_type == 'Conv':
                if self.groups > 1:
                    assert self.w.shape[0] % (self.w.shape[1] * self.groups) == 0, \
                        f'When groups({self.groups}) > 1 for {self.name}' \
                        f' out_channels({self.w.shape[0]}) must be divisible to input_channels({self.w.shape[1] * self.groups})'
                self.requires_shape = tuple(self.w.shape[-2:])
                self.scale_change = 1 / self.strides[-1]
                self.read_type = 'conv' if self.requires_shape > (1, 1) and self.scale_change == 1.0 else self.read_type
            else:
                assert tuple(self.w.shape[-2:]) == self.strides, \
                    f'Supporting conv transpose only with strides({self.strides}) = kernel({self.w.shape[-2:]}) size'
                self.scale_change = self.strides[-1]
        elif self.op_type in ('Constant', 'ConstantOfShape'):
            self.op_type = 'Constant'
            self.input_names = []
            value = get_attribute(onnx_node, "value").t
            w = np.frombuffer(value.raw_data, dtype=dtype_mapper(value.data_type))
            self.w = w[0] if len(value.dims) == 0 else w.reshape(value.dims)
        elif self.op_type == 'DepthToSpace':
            self.scale_change = get_attribute(onnx_node, "blocksize").i
        elif self.op_type == 'SpaceToDepth':
            shape = self.pattern_ops[0].constants[0].w
            shape = (shape[-3], shape[-1])
            self.requires_shape = shape
            self.scale_change = 1 / shape[-1]
        elif self.op_type == 'Resize':
            self.read_type = 'resize'
            self.sync = True
            if self.input_names[-1] in weights:
                self.scale_change = weights[self.input_names[-1]][-1]
                self.input_names = self.input_names[:-2]
        elif self.op_type == 'Repeat':
            self.w = self.pattern_ops[-1].constants[0].w
        elif self.op_type == 'Clip':
            if self.input_names[-2] in weights and self.input_names[-1] in weights:
                self.clip_min = weights[self.input_names[-2]]
                self.clip_max = weights[self.input_names[-1]]
                self.input_names = self.input_names[:-2]
        elif self.op_type == 'Slice':
            self.slice_start = 0
            self.slice_end = 0
            self.slice_axes = 0
            self.slice_steps = 0
        elif self.op_type == 'PRelu':
            self.slope = weights[self.input_names[-1]][:, 0, 0]
            self.input_names = self.input_names[:-1]
        elif self.op_type == 'LeakyRelu':
            self.slope = get_attribute(onnx_node, 'alpha').f

        self.read_shape = self.requires_shape

    def get_opts(self):
        return {opt: self.__dict__[opt] for opt in ('depth', 'read_type', 'root_op', 'out_channels')}

    def requires_synchronization(self):
        return self.requires_shape > (1, 1) or self.sync

    def __hash__(self):
        return hash((self.name, self.op_type))

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return False
        return self.name == other.name and self.op_type == other.op_type

    def post_init(self):
        if self.op_type == 'Resize':
            self.op_type = 'read'
            if len(self.constants) > 0:
                self.scale_change = self.constants[0].w[-1]
        elif self.op_type == 'Clip':
            if len(self.constants) > 0:
                self.clip_min = self.constants[0].w
                self.clip_max = self.constants[1].w
        elif self.op_type == 'Slice':
            self.slice_start = self.constants[0].w[0]
            self.slice_end = self.constants[1].w[0]
            self.slice_axes = self.constants[2].w[0]
            self.slice_steps = self.constants[3].w[0]
            assert self.slice_axes == 1, \
                f'Slice supported only for channels dimension but got channel {self.slice_axes} for op {self.name}'

    def update_channels(self):
        input_channels = [op.out_channels for op in self.input if op.op_type not in ('Constant')]

        if self.name in INPUT_NAMES:
            pass
        elif self.op_type == 'Conv':
            assert len(self.input) == 1, \
                f'Get more inputs than 1 for conv operation: {self.name} --- {self.input_names}'
            prev_channels = input_channels[0]
            assert prev_channels // self.groups == self.w.shape[1], \
                f'Input for conv {self.name} --- {prev_channels} mismatch to weight {self.w.shape}'
            self.out_channels = self.w.shape[0]
        elif self.op_type == 'ConvTranspose':
            assert len(self.input) == 1, \
                f'Get more inputs than 1 for conv transpose operation: {self.name} --- {self.input_names}'
            prev_channels = input_channels[0]
            assert prev_channels == self.w.shape[0], \
                f'Input for conv transpose {self.name} --- {prev_channels} mismatch to weight {self.w.shape}'
            self.out_channels = self.w.shape[1] * self.groups
        elif self.op_type == 'Concat':
            self.out_channels = sum([op.out_channels for op in self.input])
        elif self.op_type == 'DepthToSpace':
            assert len(self.input) == 1, \
                f'Get more inputs than 1 for DepthToSpace operation: {self.name} --- {self.input_names}'
            self.out_channels = self.input[0].out_channels // self.scale_change ** 2
        elif self.op_type == 'SpaceToDepth':
            assert len(self.input) == 1, \
                f'Get more inputs than 1 for SpaceToDepth operation: {self.name} --- {self.input_names}'
            self.out_channels = self.input[0].out_channels * int(1 / self.scale_change) ** 2
        elif self.op_type == 'Repeat':
            assert len(self.input) == 1, \
                f'Get more inputs than 1 for Repeat operation: {self.name} --- {self.input_names}'
            self.out_channels = self.input[0].out_channels * self.w[1]
        elif self.op_type == 'Slice':
            assert len(self.input) == 1, \
                f'Get more inputs than 1 for slice operation: {self.name} --- {self.input_names}'
            self.out_channels = len(range(self.slice_start, self.slice_end, self.slice_steps))
        else:
            assert all([x == input_channels[0] for x in input_channels]), \
                f'Input for {self.name} different {input_channels}'
            self.out_channels = input_channels[0]


shader_manager = ShaderManager()
ops = {op.name: op for op in [Operation(op_type=nd.op_type, name=nd.output[0], onnx_node=nd,
                                        input_names=list(nd.input), weights=weights) for nd in onnx_graph.node]}
INPUT_NAMES = []
input_ops = [op for op in ops.values() if INPUT_NAME in op.input_names]
for op in input_ops:
    name = f'{INPUT_NAME}_{op.name}'
    ops[name] = Operation(op_type='read', name=name)
    ops[name].out_channels = int(onnx_graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    op.input_names[op.input_names.index(INPUT_NAME)] = name
    INPUT_NAMES.append(name)

OUTPUT_WRITE = f'{OUTPUT_NAME}_write'
ops[OUTPUT_WRITE] = Operation(op_type='write', name=OUTPUT_WRITE, input_names=(OUTPUT_NAME,))
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


def merge_list_dicts(*dicts):
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    result = defaultdict(list)
    for k in keys:
        for d in dicts:
            result[k] += d[k]
    return result


def find_patterns(op, state, visited, pattern=None):
    if state == PatternState.INITIAL:
        visited.add(op)

    patterns, state_patterns = defaultdict(list), []
    for nxt in op.output:
        if state == PatternState.INITIAL:
            # chain of operations
            if nxt.op_type.lower() in ('mul', 'div', 'add', 'sub', 'clip', 'leakyrelu', 'prelu') \
                    + tuple(shader_manager.fun_defines):
                patterns['OpChain'].extend(find_patterns(nxt, PatternState.OP_CHAIN, visited, pattern=[nxt]))
            # silu op
            if nxt.op_type == 'Sigmoid':
                patterns['Silu'].extend(find_patterns(nxt, PatternState.SILU, visited, pattern=[nxt]))
            # SpaceToDepth
            if nxt.op_type == 'Reshape':
                patterns['SpaceToDepth'].extend(
                    find_patterns(nxt, PatternState.SPACE2DEPTH_RESHAPE1, visited, pattern=[nxt]))
            # GELU
            if nxt.op_type == 'Div' and (nxt.constants[0].w - 1.41) < 1e-2:
                patterns['Gelu'].extend(find_patterns(nxt, PatternState.GELU_DIV, visited, pattern=[nxt]))

            # Repeat channels
            if nxt.op_type == 'Expand':
                patterns['Repeat'].extend(
                    find_patterns(nxt, PatternState.REPEAT_EXPAND, visited, pattern=[nxt]))
            # try all other operations if they could be a start of pattern
            if nxt not in visited:
                patterns = merge_list_dicts(patterns, find_patterns(nxt, PatternState.INITIAL, visited))
        elif state == PatternState.OP_CHAIN:
            if nxt.op_type.lower() in ('mul', 'div', 'add', 'sub', 'clip', 'leakyrelu', 'prelu') \
                    + tuple(shader_manager.fun_defines):
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
            if nxt.op_type == 'Add' and nxt.constants[0].w == 1:
                state_patterns.extend(find_patterns(nxt, PatternState.GELU_ADD, visited, pattern=pattern + [nxt]))
        elif state == PatternState.GELU_ADD:
            if nxt.op_type == 'Mul' and set(nxt.input) == {pattern[-1], pattern[0].input[0]}:
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


visited = set()
patterns = merge_list_dicts(*[find_patterns(ops[name], PatternState.INITIAL, visited=visited) for name in INPUT_NAMES])
patterns = [(name, __pattern) for name, patterns_lst in patterns.items() for __pattern in patterns_lst]
patterns = sorted(patterns, key=lambda pair: len(pair[1]), reverse=True)
pattern_used_ops = set()
for name, __pattern in patterns:
    # print(name)
    if len(__pattern) == 0 or any([item in pattern_used_ops for item in __pattern]):
        continue
    pattern_used_ops.update(__pattern)
    # print([op.name for op in pattern])

    for op in __pattern:
        ops.pop(op.name)

    op = Operation(name=__pattern[-1].name, op_type=name, pattern_ops=__pattern)

    for el in __pattern:
        for __op in el.input:
            if __op not in __pattern:
                for idx in [idx for idx, item in enumerate(__op.output) if item == el]:
                    __op.output[idx] = op
                op.input.append(__op)

        for __op in el.output:
            if __op not in __pattern:
                for idx in [idx for idx, item in enumerate(__op.input) if item == el]:
                    __op.input[idx] = op
                op.output.append(__op)

    op.input = [*dict.fromkeys(op.input)]
    op.output = [*dict.fromkeys(op.output)]
    ops[op.name] = op


def get_scale_key(scale):
    return f"{int(scale * 10):02d}"


def get_tex_name(idx, scale):
    scale_key = get_scale_key(scale)
    return f'tex{scale_key}_{idx}'


class Texture:

    def __init__(self, begin, end, scale=1, names=None):
        self.begin = begin
        self.end = end
        self.scale = scale
        self.names = names

        self.required = set()

    def get_names(self):
        if self.names:
            return self.names
        self.names = []

        idx = self.begin
        while idx <= self.end:
            self.names.append(get_tex_name(idx // 4, self.scale))
            idx += 4 - (idx % 4)
        return self.names

    def add_requirement(self, depth):
        self.required.add(depth)

    def max_requirement(self):
        return max(self.required)


class Vec:

    def __init__(self, channels, op_idx, shape=(1, 1), base_type='float'):
        self.shape = shape
        self.op_idx = op_idx
        self.channels = channels
        self.base_type = base_type

    def get_names(self):
        names = []
        for x in range(self.shape[0]):
            names.append([])
            for y in range(self.shape[1]):
                names[x].append([])
                idx = 0
                while idx < self.channels:
                    step = min(4, self.channels - idx)
                    names[x][y].append((f'var{self.op_idx}_{x}_{y}_{idx // 4}',
                                        f'{self.base_type}{"" if step == 1 else step}'))
                    idx += step
        return names


class TextureManager:

    def __init__(self):
        self.textures = {}

    def get_textures(self, channels, depth, scale):
        # get list of textures for this scale
        scale_key = get_scale_key(scale)
        if scale_key not in self.textures:
            self.textures[scale_key] = []

        # take only texs that could be used
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


def add_between(op, first, second):
    for idx in [idx for idx, item in enumerate(first.output) if item == second]:
        first.output[idx] = op
    for idx in [idx for idx, item in enumerate(second.input) if item == first]:
        second.input[idx] = op
    op.input.append(first)
    op.output.append(second)


# pass order
sync_scales = dict()
in_channels, out_channels = defaultdict(lambda: 0), defaultdict(lambda: 0)
for op in reversed(get_backward_order(ops)):

    prev_scales = [inp.scale for inp in op.input]
    assert all([scale == prev_scales[0] for scale in prev_scales]), \
        f'All scales provided to {op.name} must match but found {prev_scales}'
    op.scale = op.scale_change * (prev_scales[0] if len(prev_scales) > 0 else op.scale)

    op.depth = max([__op.depth for __op in op.input], default=op.depth)

    if op.requires_synchronization():
        op.depth += 1
        while op.depth in sync_scales and sync_scales[op.depth] != op.scale:
            op.depth += 1
        sync_scales[op.depth] = op.scale

    op.update_channels()

if len([op for op in ops.values() if op.op_type == 'read' and op.depth == 1]) == 0:
    assert 0 not in sync_scales, f'Trying to reduce passes when zero pass requires sync {sync_scales}'
    # merge first depth without sync operations to first pass
    for op in reversed(get_backward_order(ops)):
        if op.depth > 0:
            op.depth -= 1
    for depth in list(sync_scales.keys()):
        sync_scales[depth - 1] = sync_scales[depth]
if 0 not in sync_scales:
    sync_scales[0] = 1

def fix_depths(ops, depths):
    __sync_scales = dict()
    for op in reversed(get_backward_order(ops)):
        depths[op] = max([depths[__op] for __op in op.input], default=depths[op])
        if op.requires_synchronization():
            depths[op] += 1
            while depths[op] in __sync_scales and __sync_scales[depths[op]] != op.scale:
                depths[op] += 1
            __sync_scales[depths[op]] = op.scale


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
                    old_score += round_4(inp.out_channels) // 4
                if depths[inp] != depths[__op]:
                    new_score += round_4(inp.out_channels) // 4

        if old_score > new_score:
            for __op, depth in depths.items():
                __op.depth = depth
        else:
            break

# add read/write ops between connected ops in different passes
for op in list(ops.values()):
    connections = defaultdict(list)
    for nxt in op.output:
        if op.depth != nxt.depth:
            connections[nxt.depth].append(nxt)
    if len(connections) == 0:
        continue

    write = Operation(op_type='write', name=f'{op.name}_write', input_names=(op.name,), inp=[op],
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
            read = Operation(op_type='read', name=f'{op.name}_{depth}_read', input_names=(write.name,), inp=[write],
                             output=nxt_ops, depth=depth, scale=op.scale, out_channels=op.out_channels)
            ops[read.name] = read

            for nxt in nxt_ops:
                for idx in [idx for idx, item in enumerate(nxt.input) if item == op]:
                    nxt.input[idx] = read

            # connect write and read
            write.output.append(read)

# operations order
for idx, op in enumerate(reversed(get_backward_order(ops))):
    op.priority = idx * 10
    op.root_op = op.input[0].root_op if op.root_op is None else op.root_op

# calculate scale for each  pass
min_scales = defaultdict(lambda: 1e6)
for op in reversed(get_backward_order(ops)):
    if op.op_type == 'write':
        min_scales[op.depth] = min(min_scales[op.depth], op.scale)

depths = list(sorted(min_scales.keys()))

# add extra texture for regulating run size
for depth in depths:

    if sync_scales[depth] < min_scales[depth]:
        # last op is always write and we add extra output for previous node of this write
        last_op = sorted([op for op in ops.values() if op.depth == depth], key=lambda op: op.priority)[-1].input[0]
        small_write = Operation(op_type='write', name=f'small_write_{depth}', **last_op.get_opts(),
                                priority=last_op.priority + 1, scale=sync_scales[depth], inp=[last_op])
        ops[small_write.name] = small_write
        last_op.output.append(small_write)

        # for last pass we also must add extra read/write pass in this case
        if depth == depths[-1]:
            out_write, out_op = ops[OUTPUT_WRITE], ops[OUTPUT_WRITE].input[0]
            out_write.depth += 1
            extra_read = Operation(op_type='read', name=f'extra_read_{out_write.depth}', depth=out_write.depth,
                                   out_channels=out_write.out_channels, priority=out_write.priority - 1,
                                   scale=out_write.scale, output=[out_write])
            ops[extra_read.name] = extra_read
            add_between(extra_read, out_op, out_write)

            extra_write = Operation(op_type='write', name=f'extra_write_{depth}', depth=out_op.depth,
                                    out_channels=out_op.out_channels, priority=out_op.priority + 1, scale=out_op.scale,
                                    inp=[out_op], output=[extra_read])
            ops[extra_write.name] = extra_write
            add_between(extra_write, out_op, extra_read)

            min_scales[depth + 1] = min_scales[depth]
            sync_scales[depth + 1] = min_scales[depth]

        min_scales[depth] = sync_scales[depth]

# backward propagation for requires_shape
for op in get_backward_order(ops):
    if op.op_type != 'write':
        op.out_item = Vec(op.out_channels, op_idx=op.priority, shape=(1, 1))
    for inp in op.input:
        if inp.op_type != 'write' and inp.read_shape == (1, 1):
            inp.read_shape = op.read_shape
        # do backward in read type
        if op.op_type != 'read' and inp.read_type is None:
            inp.read_type = op.read_type

    # as we have one read per pass for same output then it may have different shape outputs
    next_shapes = set([nxt.read_shape for nxt in op.output])
    if op.op_type == 'read' and len(next_shapes) >= 1:
        max_shape = max(next_shapes)
        for nxt in op.output:
            if op.read_type == nxt.read_type:
                max_shape = max(max_shape, nxt.read_shape)
            elif op.read_type == 'conv':
                max_shape = max(max_shape, tuple(dim * 2 - 1 for dim in nxt.read_shape))

        for nxt in [item for item in op.output if item.read_shape != max_shape]:
            reduce = Operation(name=f'{op.name}_{nxt.name}_reduce', op_type='reduce', priority=op.priority + 1,
                               scale=op.scale, requires_shape=max_shape, **nxt.get_opts(),
                               out_item=Vec(op.out_channels, op_idx=op.priority + 1, shape=nxt.read_shape))
            ops[reduce.name] = reduce
            add_between(reduce, op, nxt)
        op.out_item.shape = max_shape

# forward propagation for scales
for op in reversed(get_backward_order(ops)):
    input_shapes = [inp.out_item.shape for inp in op.input if inp.op_type != 'write']
    assert all([shape == input_shapes[0] for shape in input_shapes]), \
        f'All input shapes for {op.name} must match, but found {input_shapes}'

    if op.op_type != 'write':
        if op.op_type == 'read':
            # if it was not affected by sync operations then just read in appropriate scale
            if op.out_item.shape == (1, 1):
                op.out_item.shape = tuple([int(op.scale / min_scales[op.depth]) for dim in op.out_item.shape])
        # for any scale change operation it's output shape must change
        elif op.scale_change != 1:
            op.out_item.shape = tuple([int(prev_dim * op.scale_change) for prev_dim in input_shapes[0]])
        # any sync operation makes default shape as it is the run shape
        elif op.requires_shape > (1, 1):
            op.out_item.shape = op.out_item.shape
        # else propagate from input
        elif len(input_shapes) > 0:
            op.out_item.shape = input_shapes[0]


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


def prec(val):
    return f'{val:.6f}'


def type_size(tp):
    if ord('1') <= ord(tp[-1]) <= ord('9'):
        return int(tp[-1])
    else:
        return 1


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
    return values


def apply_op(expr, op, var_idx, tp, step, fun_defines):
    op_signs = {
        'Add': '+',
        'Mul': '*',
        'Div': '/',
        'Sub': '-'
    }
    if op.op_type in op_signs:
        return f"({f' {op_signs[op.op_type]} '.join([expr] + [prec(constant.w) for constant in op.constants])})"
    elif op.op_type.lower() in fun_defines:
        return f'{op.op_type.upper()}({expr})'
    elif op.op_type == 'Clip':
        return f"min(max({expr}, {prec(op.clip_min)}), {prec(op.clip_max)})"
    elif op.op_type == 'LeakyRelu':
        return f"max(0, {expr}) + {prec(op.slope)} * min(0, {expr})"
    elif op.op_type == 'PRelu':
        if len(op.slope) == 1:
            value = prec(op.slope[0])
        else:
            value = f"{tp}({', '.join([prec(item) for item in op.slope[var_idx: var_idx + step]])})"
        return f"max(0, {expr}) + {value} * min(0, {expr})"
    else:
        raise NotImplementedError(f'Not implemented applying operation of type {op.op_type}')


def conv_mul(inp_var):
    pass


def process_op(op, shader_manager):
    code = f"//{op.op_type} {op.name}\n"
    vec = op.out_item
    variables = vec.get_names() if isinstance(vec, Vec) else None
    input_vecs = [op.out_item for op in op.input]
    input_names = [var.get_names() for var in input_vecs]
    if op.op_type == 'read':
        if op.name in INPUT_NAMES:
            texs = INPUT_TEXTURE
        else:
            texs = input_vecs[0]
        tex_names = texs.get_names()
        for row_id, row in enumerate(variables):
            for column_id, column in enumerate(row):
                for var_idx, (var, tp) in enumerate(column):
                    step = min(4, vec.channels - var_idx * 4)
                    if op.read_type == 'resize':
                        pos = f"(pos + 0.5f) / ({op.scale_change} * GetInputSize())"
                        code += f"{tp} {var} = {tex_names[var_idx]}.SampleLevel({shader_manager.sampler}, {pos}, 0)" \
                                f"{f'.xyzw'[:step + 1] if step < 4 else ''};\n"
                    else:
                        # reading for scaling or for convolution
                        if op.read_type == 'conv':
                            pos = f"pos + int2({row_id - (len(variables) - 1) // 2}, {column_id - (len(row) - 1) // 2})"
                        else:
                            pos = f"int2(pos {f'* {len(variables)} ' if len(variables) > 1 else ''}+ int2({row_id}, {column_id}))"
                        # input preprocessing right after reading
                        if op.name in INPUT_NAMES:
                            code += f"{tp} {var} = load({pos});\n"
                        else:
                            code += f"{tp} {var} = {tex_names[var_idx]}[{pos}]" \
                                    f"{f'.xyzw'[:step + 1] if step < 4 else ''};\n"
    elif op.op_type == 'reduce':
        for row_id, row in enumerate(variables):
            for col_id, column in enumerate(row):
                for var_idx, (var, tp) in enumerate(column):
                    prev_vars = input_names[0]
                    prev_row, prev_col = (len(prev_vars) - 1) // 2, (len(prev_vars) - 1) // 2
                    if op.read_type == 'conv':
                        prev_row += row_id - (len(variables) - 1) // 2
                        prev_col += col_id - (len(variables) - 1) // 2
                    else:
                        prev_row, prev_col = prev_row + row_id, prev_col + col_id
                    code += f"{tp} {var} = {prev_vars[prev_row][prev_col][var_idx][0]};\n"
    elif op.op_type == 'Conv':
        variables = variables[0][0]
        for var_idx, (var, tp) in enumerate(variables):
            step = min(4, vec.channels - var_idx * 4)
            values = [prec(val) for val in op.b[4 * var_idx: 4 * var_idx + step].tolist()]
            code += f"{tp} {var} = {tp}({', '.join(values)});\n"
            for row_id in range(op.w.shape[-2]):
                for col_id in range(op.w.shape[-1]):
                    for inp_idx, (inp_var, inp_tp) in enumerate(input_names[0][row_id][col_id]):
                        if op.w.shape[1] * op.groups == op.w.shape[0] and op.groups == op.w.shape[0]:
                            # one to one depthwise convolution
                            matrix = op.w[var_idx * 4: var_idx * 4 + step:, 0, row_id, col_id]
                            matrix_values = ', '.join([prec(val) for val in matrix])
                            code += f"{var} += {inp_tp}({matrix_values}) * {inp_var};\n"
                        elif op.groups == 1:
                            # usual full convolution
                            inp_step = min(4, input_vecs[0].channels - inp_idx * 4)
                            matrix = op.w[var_idx * 4: var_idx * 4 + step, inp_idx * 4: inp_idx * 4 + inp_step,
                                     row_id, col_id]
                            matrix_values = ',\n\t'.join([', '.join([prec(val) for val in row]) for row in matrix])
                            code += f"{var} += mul({tp if type_size(tp) > 1 else f'{tp}1'}x{inp_step}(\n\t" \
                                    f"{matrix_values}" \
                                    f"), {inp_var});\n"
    elif op.op_type == 'ConvTranspose':
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                for var_idx, (var, tp) in enumerate(col):
                    step = min(4, vec.channels - var_idx * 4)
                    values = [prec(val) for val in op.b[4 * var_idx: 4 * var_idx + step]]
                    code += f"{tp} {var} = {tp}({', '.join(values)});\n"
                    for inp_idx, (inp_var, inp_tp) in enumerate(
                            input_names[0][int(row_id / op.scale_change)][int(col_id / op.scale_change)]):
                        if op.w.shape[1] * op.groups == op.w.shape[0] and op.groups == op.w.shape[0]:
                            # one to one depthwise convolution
                            matrix = op.w[var_idx * 4: var_idx * 4 + step:, 0, row_id % op.scale_change,
                                     col_id % op.scale_change]
                            matrix_values = ', '.join([prec(val) for val in matrix])
                            code += f"{var} += {inp_tp}({matrix_values}) * {inp_var};\n"
                        elif op.groups == 1:
                            # usual full convolution
                            inp_step = min(4, input_vecs[0].channels - inp_idx * 4)
                            matrix = op.w[inp_idx * 4: inp_idx * 4 + inp_step, var_idx * 4: var_idx * 4 + step,
                                     row_id % op.scale_change, col_id % op.scale_change].transpose()
                            matrix_values = ',\n\t'.join([', '.join([prec(val) for val in row]) for row in matrix])
                            code += f"{var} += mul({tp if type_size(tp) > 1 else f'{tp}1'}x{inp_step}(\n\t" \
                                    f"{matrix_values}" \
                                    f"), {inp_var});\n"
    elif op.op_type == 'Clip':
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                for var_idx, (var, tp) in enumerate(col):
                    code += f"{tp} {var} = min(max({input_names[0][row_id][col_id][var_idx][0]}, " \
                            f"{prec(op.clip_min)}), {prec(op.clip_max)});\n"
    elif op.op_type == 'Slice':
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                prev_vars = []
                for inp_idx, (inp_var, inp_tp) in enumerate(input_names[0][row_id][col_id]):
                    if 4 * inp_idx + type_size(inp_tp) >= op.slice_start and 4 * inp_idx >= op.slice_end:
                        continue
                    if 4 * inp_idx >= op.slice_start and 4 * inp_idx + type_size(inp_tp) <= op.slice_end:
                        prev_vars.append((inp_var, inp_tp))
                    elif 4 * inp_idx < op.slice_start:
                        prev_vars.extend([
                            (f"{inp_var}{f'.' + 'xyzw'[idx - 4 * inp_idx] if type_size(inp_tp) == 1 else ''}",
                             inp_tp if type_size(inp_tp) == 1 else inp_tp[:-1])
                            for idx in range(op.slice_start, 4 * inp_idx + type_size(inp_tp))])
                    elif 4 * inp_idx + type_size(inp_tp) > op.slice_end:
                        prev_vars.extend([
                            (f"{inp_var}{f'.' + 'xyzw'[idx - 4 * inp_idx] if type_size(inp_tp) == 1 else ''}",
                             inp_tp if type_size(inp_tp) == 1 else inp_tp[:-1])
                            for idx in range(4 * inp_idx, op.slice_end)])
                    else:
                        raise NotImplementedError(f"Unexpected slice case {op.name}")
                values = concat_variables(prev_vars, variables[row_id][col_id])
                for (var, var_tp), val in zip(variables[row_id][col_id], values):
                    code += f"{var_tp} {var} = {val};\n"
    elif op.op_type == 'Concat':
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                prev_vars = [var for var_names in input_names for var in var_names[row_id][col_id]]
                values = concat_variables(prev_vars, variables[row_id][col_id])
                for (var, var_tp), val in zip(variables[row_id][col_id], values):
                    code += f"{var_tp} {var} = {val};\n"
    elif op.op_type == 'Repeat':
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                prev_vars = [var for repeat_id in range(op.w[1]) for var in
                             input_names[0][row_id // op.w[-2]][col_id // op.w[-1]]]
                values = concat_variables(prev_vars, variables[row_id][col_id])
                for (var, var_tp), val in zip(variables[row_id][col_id], values):
                    code += f"{var_tp} {var} = {val};\n"
    elif op.op_type in ('Add', 'Div', 'Mul', 'Sub'):
        assert all([var.shape == vec.shape for var in input_vecs]), \
            f'All inputs to {op.op_type} {op.name} must be same shape but found {[var.shape for var in input_vecs]}'
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                for var_idx, (var, tp) in enumerate(col):
                    op_sign = {
                        'Add': '+',
                        'Mul': '*',
                        'Div': '/',
                        'Sub': '-'
                    }[op.op_type]
                    expression = f' {op_sign} '.join(
                        [var_names[row_id][col_id][var_idx][0] for var_names in input_names] +
                        [prec(constant.w) for constant in op.constants]
                    )
                    code += f"{tp} {var} = {expression};\n"
    elif op.op_type.lower() in shader_manager.fun_defines:
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                for var_idx, (var, tp) in enumerate(col):
                    code += f"{tp} {var} = {op.op_type.upper()}({input_names[0][row_id][col_id][var_idx][0]});\n"
    elif op.op_type == 'OpChain':
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                for var_idx, (var, tp) in enumerate(col):
                    chain_ops = op.pattern_ops
                    step = min(4, vec.channels - var_idx * 4)
                    op_signs = {
                        'Add': '+',
                        'Mul': '*',
                        'Div': '/',
                        'Sub': '-'
                    }
                    if chain_ops[0].op_type in op_signs:
                        result = f' {op_signs[chain_ops[0].op_type]} '.join(
                            [var_names[row_id][col_id][var_idx][0] for var_names in input_names] +
                            [prec(constant.w) for constant in chain_ops[0].constants])
                        expr = f"({result})"
                    else:
                        expr = apply_op(input_names[0][row_id][col_id][var_idx][0], chain_ops[0], var_idx, tp, step,
                                        shader_manager.fun_defines)

                    for __op in chain_ops[1:]:
                        expr = apply_op(expr, __op, var_idx, tp, step, shader_manager.fun_defines)
                    code += f"{tp} {var} = {expr};\n"
    elif op.op_type == 'SpaceToDepth':
        inverse_scale = int(1 / op.scale_change)
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                for var_idx, (var, tp) in enumerate(col):
                    values = []
                    for local_idx in range(type_size(tp)):
                        idx = 4 * var_idx + local_idx
                        x = row_id * inverse_scale + (idx // inverse_scale) % inverse_scale
                        y = col_id * inverse_scale + idx % inverse_scale
                        ch_idx = idx // (inverse_scale ** 2)
                        inp_var, inp_tp = input_names[0][x][y][ch_idx // 4]
                        values.append(f"{inp_var}{'.' + 'xyzw'[ch_idx % 4] if type_size(inp_tp) > 1 else ''}")
                    code += f"{tp} {var} = {tp}({', '.join(values)});\n"
    elif op.op_type == 'DepthToSpace':
        values = {}
        for row_id in range(len(variables)):
            values[row_id] = {}
            for col_id in range(len(variables[row_id])):
                values[row_id][col_id] = []
        for row_id, row in enumerate(input_names[0]):
            for col_id, col in enumerate(row):
                for inp_idx, (inp_var, inp_tp) in enumerate(col):
                    inp_size = type_size(inp_tp)
                    for local_idx in range(inp_size):
                        idx = 4 * inp_idx + local_idx
                        x = row_id * op.scale_change + (idx // op.scale_change) % op.scale_change
                        y = col_id * op.scale_change + idx % op.scale_change
                        values[x][y].append(f"{inp_var}{'.' + 'xyzw'[local_idx] if inp_size > 1 else ''}")
        for row_id, row in enumerate(variables):
            for col_id, col in enumerate(row):
                for var_idx, (var, tp) in enumerate(col):
                    step = min(4, vec.channels - var_idx * 4)
                    var_values = ', '.join(values[row_id][col_id][var_idx * 4: var_idx * 4 + step])
                    code += f"{tp} {var} = {tp}({var_values});\n"
    elif op.op_type == 'write' and not op.name.startswith('small_write'):
        if op.name == OUTPUT_WRITE:
            texs = OUTPUT_TEXTURE
        else:
            texs = op.out_item
        tex_names = texs.get_names()
        prev_vec = input_vecs[0]
        for row_id, row in enumerate(prev_vec.get_names()):
            for col_id, col in enumerate(row):
                for inp_idx, (inp_var, inp_tp) in enumerate(col):
                    step = min(4, prev_vec.channels - inp_idx * 4)
                    pos = f"int2(pos.x{f' * {prev_vec.shape[0]}' if prev_vec.shape[0] > 1 else ''} + {row_id}, " \
                          f"pos.y{f' * {prev_vec.shape[1]}' if prev_vec.shape[1] > 1 else ''} + {col_id})"
                    # output preprocessing before writing
                    if op.name == OUTPUT_WRITE:
                        code += f"write({inp_var}, {pos}, {op.scale});\n"
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
                        code += f"{tex_names[inp_idx]}[{pos}] = {values};\n"

    shader_manager.add_pass_text(code, op.depth)


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
print(shader)