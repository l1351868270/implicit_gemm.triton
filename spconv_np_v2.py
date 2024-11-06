'''
spconv v2.x
[SECOND:Sparsely Embedded Convolutional Detection](https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf)
[spconv](https://github.com/traveller59/spconv/blob/master/docs/spconv2_algo.pdf)
https://towardsdatascience.com/how-does-sparse-convolution-work-3257a0a8fd1
'''
import functools
import numpy as np
from typing import List, Optional
from collections import OrderedDict
import copy
from numba import jit


class NPSparseConvTensor(object):
    def __init__(self, features, indices, spatial_shape, batch_size,
                 grid=None):
        """
        Args:
            features: [num_points, num_features] feature tensor
            indices: [num_points, ndim + 1] indice tensor. batch index saved in indices[:, 0]
            spatial_shape: spatial shape of your sparse data
            batch_size: batch size of your sparse data
            grid: pre-allocated grid tensor. should be used when the volume of spatial shape
                is very large.
        """
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}
        self.grid = grid


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] *
                (kernel_size[i] - 1) - 1) // stride[i] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


class ConvOpType(object):
    kForward = 0
    kBackwardInput = 1
    kBackwardWeight = 2


class ConvMode(object):
    kCrossCorrelation = 0


class ConvProblemCommon(object):
    @staticmethod
    def implicit_gemm_mnk(self, op_type: ConvOpType, N: int, C: int, K: int, kernel_volume: int, in_prod: int, out_prod: int, mask_sparse: bool):
        if mask_sparse:
            if op_type == ConvOpType.kForward:
                return [N, K, C * kernel_volume]
            elif op_type == ConvOpType.kBackwardInput:
                return [N, C, K * kernel_volume]
            elif op_type == ConvOpType.kBackwardWeight:
                return [K, C * kernel_volume, N]
            else:
                return []
        else:
            if op_type == ConvOpType.kForward:
                return [N * out_prod, K, C * kernel_volume]
            elif op_type == ConvOpType.kBackwardInput:
                return [N * in_prod, C, K * kernel_volume]
            elif op_type == ConvOpType.kBackwardWeight:
                return [K, C * kernel_volume, N * out_prod]
            else:
                return []
            
    @staticmethod
    def conv_iwo_012_to_abc(self, op_type: ConvOpType):
        if op_type == ConvOpType.kForward:
            return [0, 1, 2]
        elif op_type == ConvOpType.kBackwardInput:
            return [2, 1, 0]
        elif op_type == ConvOpType.kBackwardWeight:
            return [1, 2, 0]
        else:
            return []
        
    @staticmethod
    def gemm_abc_012_to_iwo(self, op_type: ConvOpType):
        if op_type == ConvOpType.kForward:
            return [0, 1, 2]
        elif op_type == ConvOpType.kBackwardInput:
            return [2, 1, 0]
        elif op_type == ConvOpType.kBackwardWeight:
            return [2, 0, 1]
        else:
            return []

def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b

class ConvProblem(object):
    def __init__(self, N: int, C: int, K: int, input_dims: List[int], output_dims: List[int], ksize: List[int], padding: List[int], stride: List[int], dilation: List[int], mode=ConvMode.kCrossCorrelation, split_k_slices: int = 1, groups: int = 1):
        self.N = N
        self.C = C
        self.K = K
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.ksize = ksize
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.mode = mode
        self.split_k_slices = split_k_slices
        self.groups = groups

    def get_npq_shape(self):
        return [self.N, *self.output_dims]

    def check_npq_not_overflow(self):
        shape = self.get_npq_shape()
        return abs(functools.reduce(lambda x, y: x * y, shape, 1)) < 2**31 - 1
    
    def calc_out_dims(self, input_dims: List[int], ksize: List[int], padding: List[int], stride: List[int], dilation: List[int]):
        out = [0, 0, 0]
        for i in range(3):
            out[i] = ((input_dims[i] + padding[i] * 2 - ksize[i] * dilation[i]) // stride[i]) + 1
        return out
    
    def implicit_gemm_mnk(self, op_type: ConvOpType):
        ksize_prod = functools.reduce(lambda x, y: x * y, self.ksize, 1)
        in_prod = functools.reduce(lambda x, y: x * y, self.input_dims, 1)
        out_prod = functools.reduce(lambda x, y: x * y, self.output_dims, 1)
        return ConvProblemCommon.implicit_gemm_mnk(op_type, self.N, self.C, self.K, ksize_prod, in_prod, out_prod, False)

    def implicit_gemm_k_iterations(self, op_type: ConvOpType, tile_shape_k: int) -> int:
        ksize_prod = functools.reduce(lambda x, y: x * y, self.ksize, 1)
        in_prod = functools.reduce(lambda x, y: x * y, self.input_dims, 1)
        out_prod = functools.reduce(lambda x, y: x * y, self.output_dims, 1)
        if op_type == ConvOpType.kForward:
            return ksize_prod * div_up(div_up(self.C, self.split_k_slices), tile_shape_k)
        elif op_type == ConvOpType.kBackwardInput:
            return ksize_prod * div_up(div_up(self.K, self.split_k_slices), tile_shape_k)
        elif op_type == ConvOpType.kBackwardWeight:
            return div_up(div_up(self.N * out_prod, self.split_k_slices), tile_shape_k)
        else:
            return 0
    
    def get_input_shape(self) -> List[int]:
        return [self.N, *self.input_dims, self.C]
    
    def get_weight_shape(self) -> List[int]:
        return [self.K, *self.ksize, self.C]  
    
    def get_output_shape(self) -> List[int]:
        return [self.N, *self.output_dims, self.K]

class TensorGeneric(object):
    def __init__(self, strides: List[int]):
        self.strides = strides
    
    @staticmethod
    def from_shape(shape: List[int]):
        return TensorGeneric([shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3]])
    
    def __call__(self, indexes: List[int]):
        return indexes[3] + self.strides[0] * indexes[0] + self.strides[1] * indexes[1] + self.strides[2] * indexes[2]

    def inverse(self, index: int, out: List[int]):
        residual = index
        out[0] = int(residual // self.strides[0])
        residual = residual % self.strides[0]
        out[1] = int(residual // self.strides[1])
        residual = residual % self.strides[1]
        out[2] = int(residual // self.strides[2])
        out[3] = int(residual % self.strides[2])


class TensorGenericRS(object):
    def __init__(self, strides: List[int]):
        self.strides = strides
    
    @staticmethod
    def from_shape(shape: List[int]):
        return TensorGenericRS([shape[1] * shape[2], shape[2]])
    
    def __call__(self, indexes: List[int]):
        return indexes[2] + self.strides[0] * indexes[0] + self.strides[1] * indexes[1]

    def inverse(self, index: int, out: List[int]):
        residual = index
        out[0] = residual // self.strides[0]
        residual = residual % self.strides[0]
        out[1] = residual // self.strides[1]
        out[2] = residual % self.strides[1]

LayoutNPQ = TensorGeneric
LayoutRS = TensorGenericRS

class ConvOutLocIter(object):
    def __init__(self, problem: ConvProblem) -> None:
        self._problem = problem
        self._count = [0, 0, 0]
        self.layout_npq: LayoutNPQ = LayoutNPQ.from_shape([problem.N, problem.output_dims[0], problem.output_dims[1], problem.output_dims[2]])
        self.layout_rs: LayoutRS = LayoutRS.from_shape([problem.ksize[0], problem.ksize[1], problem.ksize[2]])

    def plus(self):
        self._count[2] += 1
        if self._count[2] < self._problem.ksize[2]:
            return self
        self._count[2] = 0
        self._count[1] += 1
        if self._count[1] < self._problem.ksize[1]:
            return self
        self._count[1] = 0
        self._count[0] += 1
        if self._count[0] < self._problem.ksize[0]:
            return self
        self._count[0] = 0
        return self
        
    def set_filter_offset(self, filter_offset: int):
        self.layout_rs.inverse(filter_offset, self._count) 


    def  nhw_to_npq(self, nhw_offset: List[int], NoStride: bool) -> List[int]:
        r_0 = self._count[0]
        h_0 = (nhw_offset[1] + self._problem.padding[0] - 
            r_0 * self._problem.dilation[0]) // (1 if NoStride else self._problem.stride[0])
        r_1 = self._count[1]
        h_1 = (nhw_offset[2] + self._problem.padding[1] - 
            r_1 * self._problem.dilation[1]) // (1 if NoStride else self._problem.stride[1])
        r_2 = self._count[2]
        h_2 = (nhw_offset[3] + self._problem.padding[2] - 
            r_2 * self._problem.dilation[2]) // (1 if NoStride else self._problem.stride[2])
        return [nhw_offset[0], h_0, h_1, h_2]
  
    def npq_to_nhw(self, npq_offset: List[int])  -> List[int]:
        r_0 = self._count[0]
        h_0 = npq_offset[1] * self._problem.stride[0] - self._problem.padding[0] + r_0 * self._problem.dilation[0]
        r_1 = self._count[1]
        h_1 = npq_offset[2] * self._problem.stride[1] - self._problem.padding[1] + r_1 * self._problem.dilation[1]
        r_2 = self._count[2]
        h_2 = npq_offset[3] * self._problem.stride[2] - self._problem.padding[2] + r_2 * self._problem.dilation[2]
        return [npq_offset[0], h_0, h_1, h_2]

    def query_npq(self, nhw_offset: List[int], npq_offset: List[int]) -> bool:
        npq_no_stride = self.nhw_to_npq(nhw_offset, True)
        npq_offset[0] = npq_no_stride[0]
        npq_offset[1] = npq_no_stride[1] // self._problem.stride[0];
        npq_offset[2] = npq_no_stride[2] // self._problem.stride[1];
        npq_offset[3] = npq_no_stride[3] // self._problem.stride[2];
        return (npq_no_stride[0] < self._problem.N) and \
               (npq_no_stride[0] >= 0) and \
               npq_offset[1] >= 0 and \
               npq_offset[1] < self._problem.output_dims[0] and \
               npq_offset[2] >= 0 and \
               npq_offset[2] < self._problem.output_dims[1] and \
               npq_offset[3] >= 0 and \
               npq_offset[3] < self._problem.output_dims[2] and \
               (not (npq_no_stride[1] % self._problem.stride[0])) and \
               (not (npq_no_stride[2] % self._problem.stride[1])) and \
               (not (npq_no_stride[3] % self._problem.stride[2]))
  
    def query_npq_no_stride(self, nhw_offset: List[int], npq_offset: List[int]) -> bool:    
        npq_offset_new = self.nhw_to_npq(nhw_offset, True)
        npq_offset[0] = npq_offset_new[0]
        npq_offset[1] = npq_offset_new[1]
        npq_offset[2] = npq_offset_new[2]
        npq_offset[3] = npq_offset_new[3]
        return (npq_offset[0] < self._problem.N) and (npq_offset[0] >= 0) and \
               npq_offset[1] >= 0 and npq_offset[1] < self._problem.output_dims[0] and \
               npq_offset[2] >= 0 and npq_offset[2] < self._problem.output_dims[1] and \
               npq_offset[3] >= 0 and npq_offset[3] < self._problem.output_dims[2]
  
    def query_nhw(self, npq_offset: List[int], nhw_offset: List[int]) -> bool:
        nhw_offset_new = self.npq_to_nhw(npq_offset)
        nhw_offset[0] = nhw_offset_new[0]
        nhw_offset[1] = nhw_offset_new[1]
        nhw_offset[2] = nhw_offset_new[2]
        nhw_offset[3] = nhw_offset_new[3]
        return (nhw_offset[0] < self._problem.N) and (nhw_offset[0] >= 0) and \
               nhw_offset[1] >= 0 and nhw_offset[1] < self._problem.input_dims[0] and \
               nhw_offset[2] >= 0 and nhw_offset[2] < self._problem.input_dims[1] and \
               nhw_offset[3] >= 0 and nhw_offset[3] < self._problem.input_dims[2]

    def query_nhw_out(self, npq_offset: List[int], nhw_offset: List[int]) -> bool:
        nhw_offset_new = self.npq_to_nhw(npq_offset)
        nhw_offset[0] = nhw_offset_new[0]
        nhw_offset[1] = nhw_offset_new[1]
        nhw_offset[2] = nhw_offset_new[2]
        nhw_offset[3] = nhw_offset_new[3]
        return (nhw_offset[0] < self._problem.N) and (nhw_offset[0] >= 0) and \
               nhw_offset[1] >= 0 and nhw_offset[1] < self._problem.output_dims[0] and \
               nhw_offset[2] >= 0 and nhw_offset[2] < self._problem.output_dims[1] and \
               nhw_offset[3] >= 0 and nhw_offset[3] < self._problem.output_dims[2]

ConvLocIter = ConvOutLocIter

class NPConvAlgo(object):
    Native = 0


def v2_generate_subm_conv_inds(indices: np.array, indice_pairs: np.array, out_inds: np.array, indice_num_per_loc: np.array, batch_size: int, input_dims: List[int], ksize: List[int], dilation: List[int]):
    assert ksize[0] % 2 == 1 and ksize[1] % 2 == 1 and ksize[2] %2 == 1
    stride = [1, 1, 1]
    padding = [(ksize[i] // 2) * dilation[i] for i in range(3)]
    kv = functools.reduce(lambda x, y: x * y, ksize, 1)
    INT_MAX = 2**31 - 1
    assert functools.reduce(lambda x, y: x * y, input_dims, 1) < INT_MAX
    problem = ConvProblem(batch_size, C, K, input_dims, input_dims, ksize, padding, stride, dilation)
    use_int32: bool = problem.check_npq_not_overflow()
    if not use_int32:
        raise ValueError("input_dims is too large")
    indices_pair_size = indice_pairs.shape[2]
    indices_pair_size_mul_RS = indices_pair_size * kv
    indice_pairs_ptr = indice_pairs.reshape(-1)
    hash = dict()
    indices_ptr = indices
    indice_in_num = indices.shape[0]
    loc_iter = ConvLocIter(problem)
    for i in range(indice_in_num):
        index = loc_iter.layout_npq(indices[i])
        hash[index] = i

    for filter_offset in range(kv // 2 + 1):
        filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size
        filter_offset_mul_indices_pair_size_1 = (kv - 1 - filter_offset) * indices_pair_size
        if filter_offset == kv // 2:
            for i in range(indice_in_num):
                indice_pairs_ptr[filter_offset_mul_indices_pair_size + i] = i
                indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i
        else:
            indices_ptr = indices
            indice_num_per_loc_ptr = indice_num_per_loc[filter_offset:]
            for i in range(indice_in_num):
                npq_offset = [0, 0, 0, 0]
                if loc_iter.query_npq_no_stride(indices_ptr[i], npq_offset):
                    index = loc_iter.layout_npq(npq_offset)
                    if index in hash:
                        old_num = indice_num_per_loc_ptr[0]
                        indice_num_per_loc_ptr[0] += 1
                        indice_pairs_ptr[filter_offset_mul_indices_pair_size + old_num] = i
                        indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = hash[index]
                        indice_pairs_ptr[filter_offset_mul_indices_pair_size_1 + old_num] = hash[index]
                        indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + old_num] = i
                # indices_ptr = indices_ptr[4:]
        loc_iter = loc_iter.plus()
    return indices.shape[0]


def v2_generate_subm_conv_inds_cpu(indices: np.array, indice_pairs: np.array, out_inds: np.array, indice_num_per_loc: np.array, 
                                batch_size: int, input_dims: List[int], ksize: List[int], dilation: List[int]):
    ndim = indices.shape[1] - 1
    assert len(input_dims) == ndim
    input_dims_ = []
    return v2_generate_subm_conv_inds(indices, indice_pairs, out_inds, indice_num_per_loc, batch_size, input_dims, ksize, dilation)


def v2_generate_conv_inds(indices: np.array, indice_pairs: np.array, out_inds: np.array, indice_num_per_loc: np.array, batch_size: int, output_dims: List[int], input_dims: List[int], ksize: List[int], stride: List[int], padding: List[int], dilation: List[int], transposed: bool):
    kv = functools.reduce(lambda x, y: x * y, ksize, 1) 
    problem = ConvProblem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation)
    use_int32 = problem.check_npq_not_overflow()
    if not use_int32:
        raise ValueError("input_dims is too large")
    num_act = 0
    loc_iter = ConvLocIter(problem)
    indices_pair_size = indice_pairs.shape[2]
    indices_pair_size_mul_RS = indices_pair_size * kv
    indice_pairs_ptr = indice_pairs.reshape(-1)
    hash = dict()
    indices_ptr = indices
    out_inds_ptr = out_inds.reshape(-1)
    INT_MAX = 2**31 - 1
    assert functools.reduce(lambda x, y: x * y, input_dims, 1) < INT_MAX, 'kernel volume must smaller than max value of int32'
    indice_in_num = indices.shape[0]
    hashval = 0
    for filter_offset in range(kv):
        filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size
        indices_ptr = indices
        indice_num_per_loc_ptr = indice_num_per_loc[filter_offset:]
        for i in range(indice_in_num):
            npq_offset = [0, 0, 0, 0]
            valid = loc_iter.query_npq(indices_ptr[i], npq_offset)
            if valid:
                index = loc_iter.layout_npq(npq_offset)
                if index not in hash.keys():
                    hashval = num_act
                    num_act += 1
                    hash[index] = hashval
                    for k in range(4):
                        out_inds_ptr[k] = npq_offset[k]
                    out_inds_ptr = out_inds_ptr[4:]
                else:
                    hashval = hash[index]
                indice_pairs_ptr[filter_offset_mul_indices_pair_size + indice_num_per_loc_ptr[0]] = i
                indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + indice_num_per_loc_ptr[0]] = hashval
                indice_num_per_loc_ptr[0] += 1
            # indices_ptr += 4
        loc_iter = loc_iter.plus()
    return num_act

def v2_generate_conv_inds_cpu(indices: np.array, indice_pairs: np.array, out_inds: np.array, indice_num_per_loc: np.array, 
                                batch_size: int, output_dims: List[int], input_dims: List[int], ksize: List[int], stride: List[int], padding: List[int], dilation: List[int], transposed: bool):
    ndim = indices.shape[1] - 1
    assert len(output_dims) == ndim and len(input_dims) == ndim and len(ksize) == ndim and \
           len(stride) == ndim and len(padding) == ndim and len(dilation) == ndim, f"your params size not equal to {ndim}"
    
    return v2_generate_conv_inds(indices, indice_pairs, out_inds, indice_num_per_loc, 
                                 batch_size, output_dims, input_dims, 
                                 ksize, stride, padding, dilation, transposed)


def v2_get_indice_pairs(indices: np.array,
                     batch_size: int,
                     spatial_shape: List[int],
                     algo: NPConvAlgo,
                     ksize: List[int],
                     stride: List[int],
                     padding: List[int],
                     dilation: List[int],
                     out_padding: List[int],
                     subm: bool = False,
                     transpose: bool = False,
                     num_out_act_bound: int = -1):
    kv: int = functools.reduce(lambda x, y: x * y, ksize, 1)
    if not subm:
        out_shape = get_conv_output_size(spatial_shape, ksize, stride, padding, dilation)
    else:
        out_shape = spatial_shape
    
    if any([x == 0 for x in out_shape]):
        raise ValueError(
            f"your out spatial shape {out_shape} reach zero!!! input shape: {spatial_shape}"
        )

    spatial_volume = functools.reduce(lambda x, y: x * y, out_shape, 1)
    pair = np.full((2, kv, indices.shape[0]), -1, dtype=indices.dtype)
    indice_num_per_loc = np.zeros((kv), dtype=indices.dtype)
    if subm:
        out_inds = np.copy(indices)
        v2_generate_subm_conv_inds_cpu(indices, pair, out_inds, indice_num_per_loc, batch_size, spatial_shape, ksize, dilation)
    else:
        out_inds = np.empty((kv * indices.shape[0], indices.shape[1]), dtype=indices.dtype)
        num_act_out = v2_generate_conv_inds_cpu(indices, pair, out_inds, indice_num_per_loc, batch_size, out_shape, spatial_shape, ksize, stride, padding, dilation, transpose)
        out_inds = out_inds[:num_act_out]
    return out_inds, pair, indice_num_per_loc


def v2_gather_cpu(out: np.array, in_: np.array, inds: np.array):
    nhot = inds.shape[0]
    channel = in_.shape[1]
    buffer_data = out
    features_data = in_
    for i in range(nhot):
        buffer_data[i] = features_data[inds[i]]


def v2_scatter_add_cpu(out: np.array, in_: np.array, inds: np.array):
    nhot = inds.shape[0]
    channel = in_.shape[1]
    indices_data = inds
    buffer_data = in_
    features_data = out

    for i in range(nhot):
        for j in range(channel):
            if (indices_data[i] < 0) :
                raise ValueError(f'indices[{i}]: {indices_data[i]} < 0')
            features_data[indices_data[i]][j] += buffer_data[i][j]


def v2_indice_conv(features: np.array, filters: np.array, indice_pairs: np.array, indice_pair_num: np.array, num_activate_out: int, subm: bool, algo: int):
    kv_dim = 1
    out_channel = filters.shape[0]
    filters = filters.reshape(out_channel, -1, filters.shape[-1])
    kv = filters.shape[1]
    filter_shape_per_kv = [out_channel, filters.shape[-1]]
    kv_center = kv // 2
    
    if subm:
        out_features = np.empty((features.shape[0], out_channel),
                                           dtype=features.dtype)
        np.matmul(features, filters[:, kv_center].T, out=out_features)
    else:
        out_features = np.zeros((num_activate_out, out_channel), dtype=features.dtype)

    indice_pair_num_cpu = indice_pair_num
    indice_pair_num_cpu_ptr = indice_pair_num_cpu
    maxnhot = 0
    all_zero = True
    for i in range(kv):
        if indice_pair_num_cpu_ptr[i] != 0 :
            indice_pair_num_cpu_ptr[i] = min(indice_pair_num_cpu_ptr[i], int(indice_pairs.shape[2]))
            all_zero = False
            maxnhot = max(maxnhot, indice_pair_num_cpu_ptr[i])
        
    if subm and all_zero :
      return
    
    inited = subm
    a = features
    c = out_features
    pair_in = indice_pairs[0]
    pair_out = indice_pairs[1]

    inp_buffer = np.empty((maxnhot, features.shape[1]), dtype=features.dtype)
    out_buffer = np.empty((maxnhot, out_features.shape[1]), dtype=out_features.dtype)
    for i in range(kv):
        nhot = indice_pair_num_cpu_ptr[i]
        if subm and i == kv_center:
            continue
        if subm and i > kv_center:
            nhot = indice_pair_num_cpu_ptr[kv - i - 1]
        if nhot <= 0:
            continue
        inp_indices = pair_in[i][:nhot]
        out_indices = pair_out[i][:nhot]
        v2_gather_cpu(inp_buffer, a, inp_indices)
        filters_i = filters[:, i]
        filters_cur = filters_i.T
        np.matmul(inp_buffer[:nhot], filters_cur, out=out_buffer[:nhot])
        v2_scatter_add_cpu(c, out_buffer, out_indices)
    return out_features

def v2_sparse_convolution(input: NPSparseConvTensor, add_input: Optional[NPSparseConvTensor], weight: np.array, ndim: int, in_channels: int, out_channels: int,
                 kernel_size: List[int]=[3, 3, 3], stride: List[int]=[1, 1, 1], padding: List[int]=[0, 0, 0], 
                 dilation: List[int]=[1, 1, 1], groups: int = 1, bias: bool = True, subm: bool = False,
                 output_padding: List[int]=[0, 0, 0], transposed: bool = False, inverse: bool = False, 
                 indice_key: Optional[str]=None, algo: Optional[NPConvAlgo] = NPConvAlgo.Native, fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,  act_type = None, act_alpha: float = 0, act_beta: float = 0, large_kernel_fast_algo: bool = False,
                 name=None, device=None, dtype=None):
    features = input.features
    indices = input.indices
    spatial_shape = input.spatial_shape
    batch_size = input.batch_size
    if not subm:
        out_spatial_shape = get_conv_output_size(spatial_shape, kernel_size, stride, padding, dilation)
    else:
        out_spatial_shape = spatial_shape

    conv1x1 = np.prod(kernel_size) == 1
    if conv1x1:
        features = np.matmul(input.features, weight.reshape(in_channels, out_channels))
        if bias is not None:
            features += bias
        out_tensor = NPSparseConvTensor(features, input.indices, input.spatial_shape, input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor
    outids, indice_pairs, indice_pair_num = v2_get_indice_pairs(
        indices, batch_size, spatial_shape, algo,
        kernel_size, stride, padding, 
        dilation, output_padding, subm,
        transposed)
    if subm:
        out_features = v2_indice_conv(features, weight, indice_pairs, indice_pair_num, outids.shape[0], subm=True, algo=algo)
    else:
        out_features = v2_indice_conv(features, weight, indice_pairs, indice_pair_num, outids.shape[0], subm=False, algo=algo)
    output_tensor = NPSparseConvTensor(out_features, outids, out_spatial_shape, batch_size)
    return output_tensor


def v2_subm_conv3d(input: NPSparseConvTensor, add_input: Optional[NPSparseConvTensor], weight: np.array, in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo: Optional[NPConvAlgo]=NPConvAlgo.Native, 
                fp32_accum: Optional[bool]=None, large_kernel_fast_algo: bool=False, name=None):
    return v2_sparse_convolution(input, add_input, weight, 3, in_channels, out_channels,
                                 kernel_size, stride = stride, padding = padding, 
                                 dilation = dilation, groups = groups, bias = bias, subm = True,
                                 indice_key = indice_key, algo = algo, fp32_accum = fp32_accum,
                                 large_kernel_fast_algo = large_kernel_fast_algo, name=name)

def v2_sparse_conv3d(input: NPSparseConvTensor, add_input: Optional[NPSparseConvTensor], weight: np.array, in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo: Optional[NPConvAlgo]=NPConvAlgo.Native, 
                fp32_accum: Optional[bool]=None, large_kernel_fast_algo: bool=False, name=None):
    return v2_sparse_convolution(input, add_input, weight, 3, in_channels, out_channels,
                                 kernel_size, stride = stride, padding = padding, 
                                 dilation = dilation, groups = groups, bias = bias, subm = False,
                                 indice_key = indice_key, algo = algo, fp32_accum = fp32_accum,
                                 large_kernel_fast_algo = large_kernel_fast_algo, name=name)

if __name__ == '__main__':

    # N = 1; C = 64; D = 56; H = 56; W = 56; K = 64; T = 1; R = 1; S = 1; pad_d = 0; pad_h = 0; pad_w = 0; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1
    N = 1; C = 16; D = 9; H = 9; W = 9; K = 16; T = 3; R = 3; S = 3; pad_d = 0; pad_h = 0; pad_w = 0; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1
    # N = 1; C = 16; D = 9; H = 9; W = 9; K = 16; T = 3; R = 3; S = 3; pad_d = 1; pad_h = 1; pad_w = 1; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1

    # N = 1; C = 16; D = 9; H = 9; W = 9; K = 16; T = 1; R = 1; S = 1; pad_d = 0; pad_h = 0; pad_w = 0; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1
    
    w = np.random.randn(K, C, T, R, S).astype(np.float32)
    num_ = D * H * W
    indices_d = np.random.randint(0, D, (num_,), dtype=np.int32)
    indices_h = np.random.randint(0, H, (num_,), dtype=np.int32)
    indices_w = np.random.randint(0, W, (num_,), dtype=np.int32)
    indices = np.stack([indices_d, indices_h, indices_w], axis=0, dtype=np.int32)
    indices = np.unique(indices, axis=1)
    np.random.shuffle(indices.T)
    sp_indices = np.ascontiguousarray(indices.T, dtype=np.int32)
    sp_indices = np.concatenate((np.zeros((sp_indices.shape[0], 1), dtype=np.int32), sp_indices), axis=1, dtype=np.int32)
    num_ = len(sp_indices)
    values = np.random.randn(num_, C).astype(np.float32)
    spatial_shape = [D, H, W]

    v2_tl_w = np.ascontiguousarray(w.transpose(0, 2, 3, 4, 1))
    v2_tl_x = NPSparseConvTensor(values, sp_indices, spatial_shape, N)
    v2_tl_subm_y = v2_subm_conv3d(v2_tl_x, None, v2_tl_w, C, K, [T, R, S], stride=[stride_d, stride_h, stride_w],
                                  padding=[pad_d, pad_h, pad_w], dilation=[dila_d, dila_h, dila_w], bias=False)
    v2_tl_sparse_y = v2_sparse_conv3d(v2_tl_x, None, v2_tl_w, C, K, [T, R, S], stride=[stride_d, stride_h, stride_w],
                                  padding=[pad_d, pad_h, pad_w], dilation=[dila_d, dila_h, dila_w], bias=False)

    check = True
    if check:
        '''
        only support spconv 2.x check
        pip install spconv_cu129==2.3.6
        conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
        '''
        import torch

        import spconv.pytorch as spconv
        from spconv.core import ConvAlgo
        
        # get_indice_pairs check
        tl_subm_outids, tl_subm_indice_pairs, tl_subm_indice_pair_num = v2_get_indice_pairs(sp_indices, N, spatial_shape, NPConvAlgo.Native, [T, R, S], [stride_d, stride_h, stride_w], [pad_d, pad_h, pad_w], [dila_d, dila_h, dila_w], [0, 0, 0], True, False, -1)
        tl_sparse_outids, tl_sparse_indice_pairs, tl_sparse_indice_pair_num = v2_get_indice_pairs(sp_indices, N, spatial_shape, NPConvAlgo.Native, [T, R, S], [stride_d, stride_h, stride_w], [pad_d, pad_h, pad_w], [dila_d, dila_h, dila_w], [0, 0, 0], False, False, -1)

        sp_subm_outids, sp_subm_indice_pairs, sp_subm_indice_pair_num = spconv.ops.get_indice_pairs(
                            torch.tensor(sp_indices).int(),
                            N,
                            spatial_shape,
                            ConvAlgo.Native,
                            [T, R, S],
                            [stride_d, stride_h, stride_w],
                            [pad_d, pad_h, pad_w],
                            [dila_d, dila_h, dila_w],
                            [0, 0, 0],
                            True,
                            False,
                            -1)
        sp_sparse_outids, sp_sparse_indice_pairs, sp_sparse_indice_pair_num = spconv.ops.get_indice_pairs(
                            torch.tensor(sp_indices).int(),
                            N,
                            spatial_shape,
                            ConvAlgo.Native,
                            [T, R, S],
                            [stride_d, stride_h, stride_w],
                            [pad_d, pad_h, pad_w],
                            [dila_d, dila_h, dila_w],
                            [0, 0, 0],
                            False,
                            False,
                            -1)
        sp_w = np.asanyarray(w.transpose(0, 2, 3, 4, 1))

        assert np.allclose(sp_subm_outids.detach().cpu().numpy(), tl_subm_outids, atol=1e-4, rtol=1e-4)
        assert np.allclose(sp_subm_indice_pairs.detach().cpu().numpy(), tl_subm_indice_pairs, atol=1e-4, rtol=1e-4)
        assert np.allclose(sp_subm_indice_pair_num.detach().cpu(), tl_subm_indice_pair_num, atol=1e-4, rtol=1e-4)
        assert np.allclose(sp_sparse_indice_pair_num.detach().cpu().numpy(), tl_sparse_indice_pair_num, atol=1e-4, rtol=1e-4)
        assert np.allclose(sp_sparse_indice_pairs.detach().cpu().numpy(), tl_sparse_indice_pairs, atol=1e-4, rtol=1e-4)

        # sparse tensor check
        sp_x = spconv.SparseConvTensor(torch.as_tensor(values, device='cpu').float(), torch.as_tensor(sp_indices).int(), spatial_shape, N) # [N, C, D, H, W]
        coo_x = torch.sparse_coo_tensor(torch.as_tensor(indices, device='cpu').int(), torch.as_tensor(values, device='cpu').float(), (D, H, W, C)) # [D, H, W, C]
        assert torch.allclose(sp_x.dense()[0], coo_x.to_dense().permute(3, 0, 1, 2), atol=1e-4, rtol=1e-4)

        # SubMConv3d check
        sp_subm_conv3d = spconv.SubMConv3d(C, K, [T, R, S], stride=[stride_d, stride_h, stride_w], padding=[pad_d, pad_h, pad_w], dilation=[dila_d, dila_h, dila_w], bias=False, indice_key='sp_subm_conv3d', algo = ConvAlgo.Native)
        sp_subm_conv3d.weight.data = torch.as_tensor(sp_w, device='cpu').float()
        sp_subm_y = sp_subm_conv3d(sp_x)
        sp_subm_y_indice_pair_num = sp_subm_y.indice_dict["sp_subm_conv3d"].indice_pair_num
        sp_subm_y_indice_pairs = sp_subm_y.indice_dict["sp_subm_conv3d"].indice_pairs
        assert np.allclose(sp_subm_y_indice_pair_num.detach().cpu().numpy(), tl_subm_indice_pair_num, atol=1e-4, rtol=1e-4)
        if not np.allclose(sp_subm_y_indice_pairs.detach().cpu().numpy(), tl_subm_indice_pairs, atol=1e-4, rtol=1e-4):
            print(f'spconv.SubMConv3d and subm_conv3d differ')
            print(f'sp_subm_y_indice_pairs: {sp_subm_y_indice_pairs.detach().cpu()}')
            print(f'tl_subm_indice_pairs: {tl_subm_indice_pairs}')
        if not np.allclose(sp_subm_y.features.detach().cpu().numpy(), v2_tl_subm_y.features, atol=1e-4, rtol=1e-4):
            print(f'spconv.SubMConv3d and subm_conv3d differ')
            print(f'sp_subm_y: {sp_subm_y.features}')
            print(f'v2_tl_subm_y: {v2_tl_subm_y.features}')

        # Conv3d check
        sp_sparseconv3d = spconv.SparseConv3d(C, K, [T, R, S], stride=[stride_d, stride_h, stride_w], padding=[pad_d, pad_h, pad_w], dilation=[dila_d, dila_h, dila_w], bias=False, indice_key='sp_sparseconv3d', algo = ConvAlgo.Native)
        sp_sparseconv3d.weight.data = torch.as_tensor(sp_w, device='cpu').float()
        sp_sparse_y = sp_sparseconv3d(sp_x)
        sp_sparse_y_indice_pair_num = sp_sparse_y.indice_dict["sp_sparseconv3d"].indice_pair_num
        sp_sparse_y_indice_pairs = sp_sparse_y.indice_dict["sp_sparseconv3d"].indice_pairs
        assert np.allclose(sp_sparse_y_indice_pair_num.detach().cpu().numpy(), tl_sparse_indice_pair_num, atol=1e-4, rtol=1e-4)
        if not np.allclose(sp_sparse_y_indice_pairs.detach().cpu().numpy(), tl_sparse_indice_pairs, atol=1e-4, rtol=1e-4):
            print(f'spconv.SubMConv3d and v2_subm_conv3d differ')
            print(f'sp_subm_y_indice_pairs: {sp_subm_y_indice_pairs.detach().cpu()}')
            print(f'tl_subm_indice_pairs: {tl_subm_indice_pairs}')
        if not np.allclose(sp_sparse_y.features.detach().cpu().numpy(), v2_tl_sparse_y.features, atol=1e-4, rtol=1e-4):
            print(f'spconv.SparseConv3d and v2_sparse_conv3d differ')
            print(f'sp_sparse_y: {sp_sparse_y.features}')
            print(f'v2_tl_sparse_y: {v2_tl_sparse_y.features}')


        # torch.nn.Conv3d check
        conv3d = torch.nn.Conv3d(C, K, (T, R, S), stride=(stride_d, stride_h, stride_w), padding=(pad_d, pad_h, pad_w), dilation=(dila_d, dila_h, dila_w), bias=False).float()
        conv3d.weight.data = torch.as_tensor(w)
        torch_features = conv3d(coo_x.to_dense().permute(3, 0, 1, 2).reshape(1, C, D, H, W))
        if not torch.allclose(sp_sparse_y.dense(), torch_features, atol=1e-4, rtol=1e-4):
            assert torch.allclose(sp_sparseconv3d.weight.data, conv3d.weight.data.permute(2, 3, 4, 0, 1), atol=1e-4, rtol=1e-4)
            assert torch.allclose(sp_x.dense(), coo_x.to_dense().permute(3, 0, 1, 2).reshape(1, C, D, H, W), atol=1e-4, rtol=1e-4)
            print(sp_sparse_y.dense())
            print(torch_features)




