'''
[SECOND:Sparsely Embedded Convolutional Detection](https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf)
[spconv](https://github.com/traveller59/spconv/blob/master/docs/spconv2_algo.pdf)
https://towardsdatascience.com/how-does-sparse-convolution-work-3257a0a8fd1
'''
import functools
import numpy as np
from typing import List
from collections import OrderedDict
import copy


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


def get_valid_out_pos(input_pos, kernel_size, stride, padding, dilation, out_spatial_shape, out):
    ndim = 3
    lowers = [-1 for _ in range(ndim)]
    uppers = [-1 for _ in range(ndim)]
    counter = [-1 for _ in range(ndim)]
    counter_size = [-1 for _ in range(ndim)]
    point_counter = 0
    val = 0
    num_points = 1
    m = 0
    offset = 0
    valid = False

    for i in range(ndim):
        lowers[i] = (input_pos[i] - (kernel_size[i] - 1) * dilation[i] - 1 + stride[i] + padding[i]) // stride[i]
        uppers[i] = (input_pos[i] + padding[i]) // stride[i]
    for i in range(ndim):
        counter_size[i] = ((uppers[i] - lowers[i]) // dilation[i] + 1)
        num_points *= counter_size[i]
    for i in range(ndim):
        counter[i] = 0
    for i in range(num_points):
        valid = True
        m = 1
        offset = 0
        for j in range(ndim - 1, -1, -1):
            val = uppers[j] - counter[j] * dilation[j]
            out[point_counter * (ndim + 1) + j] = val
            if (val < 0 or (val > out_spatial_shape[j] - 1)):
                valid = False
            offset += m * (input_pos[j] - val * stride[j] + padding[j]) // dilation[j]
            m *= kernel_size[j]

        out[point_counter * (ndim + 1) + ndim] = offset
        if valid:
          point_counter += 1
        counter[ndim - 1] += 1
        for c in range(ndim - 1, -1, -1):
            if counter[c] == counter_size[c] and c > 0:
                counter[c - 1] += 1
                counter[c] = 0
    return point_counter

def row_array_idx(indices: List[int], shape: List[int]):
    ndim = len(shape)
    offset = 0
    m = 1
    for i in range(ndim - 1, -1, -1):
        offset += m * indices[i]
        m *= shape[i]
    return offset

def get_indice_pairs_subm(indices_in: np.ndarray,
                          grids_out: np.ndarray,
                          indice_pairs: np.ndarray,
                          indice_num: np.ndarray,
                          kernel_size: List[int],
                          stride: List[int],
                          padding: List[int],
                          dilation: List[int],
                          out_spatial_shape: List[int]):
    ndim = len(kernel_size)
    num_act = 0
    num_act_in = indices_in.shape[0]
    batch_idx = 0
    spatial_volume = functools.reduce(lambda x, y: x * y, out_spatial_shape, 1)
    kernel_volume = functools.reduce(lambda x, y: x * y, kernel_size, 1)
    hash = dict()
    for i in range(num_act_in):
        index = row_array_idx(indices_in[i, 1:].tolist(), out_spatial_shape) \
              + spatial_volume * int(indices_in[i, 0]) # batch index
        hash[index] = i
    index = 0
    num_valid_points = 0
    valid_points = [0 for _ in range(kernel_volume * (ndim + 1))]
    for j in range(num_act_in):
        num_valid_points = get_valid_out_pos(indices_in[j, 1:].tolist(), kernel_size, stride, padding, dilation, out_spatial_shape, valid_points)
        for i in range(num_valid_points):
            point_ptr = valid_points[i * (ndim + 1):]
               
            offset = point_ptr[ndim]
            index = row_array_idx(point_ptr, out_spatial_shape) \
                    + spatial_volume * int(indices_in[j, 0]) # batch index
            if index in hash.keys():
                old_offset = copy.deepcopy(indice_num[offset])
                indice_num[offset] += 1
                indice_pairs[0, offset, old_offset] = j
                indice_pairs[1, offset, old_offset] = hash[index]
                
    return num_act_in

def create_submconv_indice_pair_cpu(
    indices_in: np.ndarray, grids_out: np.ndarray, indice_pairs: np.ndarray,
    indice_num: np.ndarray, kernel_size: List[int],
    stride: List[int], padding: List[int],
    dilation: List[int], out_spatial_shape: List[int],
    transpose: bool, reset_grid: bool, use_hash: bool):
    ndim = len(out_spatial_shape)
    num_act_in = indices_in.shape[0]
    batch_size = grids_out.shape[0]
    kernel_volume = indice_num.shape[0]
    if num_act_in == 0:
        return 0
    num_act_in = get_indice_pairs_subm(indices_in, grids_out, indice_pairs, indice_num, kernel_size, stride, padding, dilation, out_spatial_shape)
    
    return num_act_in


def get_indice_pairs_conv(indices_in: np.ndarray,
                          indices_out: np.ndarray,
                          grids_out: np.ndarray,
                          indice_pairs: np.ndarray,
                          indice_num: np.ndarray,
                          kernel_size: List[int], stride: List[int],
                          padding: List[int], dilation: List[int],
                          out_spatial_shape: List[int]):
    ndim = len(kernel_size)
    num_act = 0
    num_act_in = indices_in.shape[0]
    batch_idx = 0
    spatial_volume = functools.reduce(lambda x, y: x * y, out_spatial_shape, 1)
    kernel_volume = functools.reduce(lambda x, y: x * y, kernel_size, 1)
    num_valid_points = 0
    valid_points = [0 for _ in range(kernel_volume * (ndim + 1))]
    hash = dict()
    for j in range(num_act_in):
        batch_idx = indices_in[j, 0]
        num_valid_points = get_valid_out_pos(indices_in[j, 1:].tolist(), kernel_size, stride, padding, dilation, out_spatial_shape, valid_points)
        for i in range(num_valid_points):
            point_ptr = valid_points[i * (ndim + 1):]
            offset = point_ptr[ndim]
            index = row_array_idx(point_ptr, out_spatial_shape) \
                    + spatial_volume * int(batch_idx)
            if index not in hash.keys():
                for k in range(1, ndim + 1):
                    indices_out[num_act, k] = point_ptr[k - 1]
                indices_out[num_act, 0] = batch_idx
                hash_val = num_act
                num_act += 1
                hash[index] = hash_val
            else:
                hash_val = hash[index]
            indice_pairs[0, offset, indice_num[offset]] = j
            indice_pairs[1, offset, indice_num[offset]] = hash_val
            indice_num[offset] += 1
    return num_act

def create_conv_indice_pair_cpu(indices_in: np.ndarray, indices_out: np.ndarray, grids_out: np.ndarray,
                                indice_pairs: np.ndarray, indice_num: np.ndarray,
                                kernel_size: List[int], stride: List[int],
                                padding: List[int], dilation: List[int],
                                out_spatial_shape: List[int], transpose: bool, reset_grid: bool,
                                use_hash: bool):
    ndim = len(out_spatial_shape)
    num_act_in = indices_in.shape[0]
    batch_size = grids_out.shape[0]
    kernel_volume = indice_num.shape[0]
    if num_act_in == 0:
        return 0
    
    num_act_in = get_indice_pairs_conv(indices_in, indices_out, grids_out, indice_pairs, indice_num, kernel_size, stride, padding, dilation, out_spatial_shape)
    return num_act_in


# Adapted from https://github.com/traveller59/spconv/blob/v2.3.6/spconv/pytorch/ops.py#L132
def get_indice_pairs(indices: np.ndarray,
                     batch_size: int,
                     spatial_shape: List[int],
                    #  algo: ConvAlgo,
                     ksize: List[int],
                     stride: List[int],
                     padding: List[int],
                     dilation: List[int],
                     out_padding: List[int],
                     subm: bool = False,
                     transpose: bool = False,
                     use_hash=False):
    ndim = len(ksize)
    num_act = indices.shape[0]
    coor_dim = indices.shape[1] - 1
    if not subm:
        out_shape = get_conv_output_size(spatial_shape, ksize, stride, padding, dilation)
    else:
        out_shape = spatial_shape
    assert ndim == coor_dim
    assert len(out_shape) == coor_dim
    assert len(stride) == coor_dim
    assert len(padding) == coor_dim
    assert len(dilation) == coor_dim
    assert len(out_padding) == coor_dim

    kernel_volume: int = functools.reduce(lambda x, y: x * y, ksize, 1)
    assert kernel_volume <= 4096

    output_volume: int = functools.reduce(lambda x, y: x * y, out_shape, 1)
    INT_MAX = 2**31 - 1
    assert batch_size * output_volume <= INT_MAX
    
    spatial_volume = functools.reduce(lambda x, y: x * y, spatial_shape, 1)
    indice_pairs = np.full((2, kernel_volume, num_act), -1, dtype=indices.dtype)
    indice_num = np.zeros((kernel_volume), dtype=indices.dtype)
    grid_size = batch_size * output_volume
    if use_hash:
        grid_size = batch_size
    grid_out = np.full((grid_size, ), -1, dtype=indices.dtype)
    grid_out = grid_out.reshape(batch_size, -1)
    if subm:
        padding = [ksize[i] // 2 for i in range(ndim)]
        stride = [1 for i in range(ndim)]
    if subm:
        num_act_out = create_submconv_indice_pair_cpu(indices, grid_out, indice_pairs, indice_num, ksize, stride, padding, 
                                    dilation, out_shape, transpose, False, use_hash)
        return indices, indice_pairs, indice_num
    else:
        indice_pair_unique = np.full((indice_pairs.size // 2 + 1, ), INT_MAX, dtype=indices.dtype)
        out_inds = np.zeros((num_act * kernel_volume, coor_dim + 1), dtype=indices.dtype)
        num_act_out = create_conv_indice_pair_cpu(indices, out_inds, grid_out, indice_pairs, indice_num, ksize, stride, 
                                                  padding, dilation, out_shape, transpose, False, use_hash)
        return out_inds[:num_act_out], indice_pairs, indice_num


def sparse_gather_cpu(buffer: np.ndarray, features: np.ndarray, indices: np.ndarray, size):
    num_planes = features.shape[1]
    buffer_data = buffer
    features_data = features
    for i in range(size):
        buffer_data[i] = features_data[indices[i]]
        

def sparse_scatter_add_cpu(buffer: np.ndarray, out_features: np.ndarray, indices: np.ndarray, size):
    num_planes = out_features.shape[1]
    buffer_data = buffer
    features_data = out_features

    for i in range(size):
        for j in range(num_planes):
            if (indices[i] < 0) :
                raise ValueError(f'indices[{i}]: {indices[i]} < 0')
            features_data[indices[i]][j] += buffer_data[i][j]


def indice_conv(features: np.ndarray, filters: np.ndarray, indice_pairs: np.ndarray, indice_pair_num: np.ndarray, num_act_out, subm=True, algo=0):
    kernel_volume = indice_pair_num.shape[0]
    ndim = len(filters.shape) - 2
    num_in_planes = features.shape[1]
    num_out_planes = filters.shape[ndim + 1]
    output = np.zeros((num_act_out, num_out_planes), dtype=features.dtype)
    filters = filters.reshape((-1, num_in_planes, num_out_planes))
    indice_pair_max_offset = kernel_volume // 2
    indice_pair_max_size = num_act_out
    if subm:
        np.matmul(features, filters[indice_pair_max_offset], out=output)
        indice_pair_max_size = indice_pair_num[:indice_pair_max_offset].max().item()
        if indice_pair_max_size == 0:
            return output
    else:
        indice_pair_max_size = indice_pair_num[:kernel_volume].max().item()

    input_buffer = np.zeros((indice_pair_max_size, num_in_planes), dtype=features.dtype)
    output_buffer = np.zeros((indice_pair_max_size, num_out_planes), dtype=features.dtype)
    total_gather_time = 0.0
    total_gemm_time = 0.0
    total_sadd_time = 0.0
    for i in range(kernel_volume):
        nhot = indice_pair_num[i]
        if nhot <= 0 or (subm and i == indice_pair_max_offset):
            continue
        
        output_buffer_blob = np.asarray(output_buffer[:nhot])
        input_buffer_blob = np.asarray(input_buffer[:nhot])
        sparse_gather_cpu(input_buffer_blob, features, indice_pairs[0][i], nhot)
        np.matmul(input_buffer_blob, filters[i], out=output_buffer_blob, dtype=features.dtype)
        sparse_scatter_add_cpu(output_buffer_blob, output, indice_pairs[1][i], nhot)
    return output

# def 

def sparse_convolution(input: NPSparseConvTensor, weight: np.ndarray, ndim, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, subm=False,
                 output_padding=[0, 0, 0], transposed=False, inverse=False, indice_key=None, fused_bn=False, use_hash=False, algo=0):
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
    outids, indice_pairs, indice_pair_num = get_indice_pairs(
        indices,
        batch_size,
        spatial_shape,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
        subm,
        transposed,
        use_hash=use_hash)
    
    if subm:
        out_features = indice_conv(features, weight, indice_pairs, indice_pair_num, outids.shape[0], subm=True, algo=algo)
    else:
        out_features = indice_conv(features, weight, indice_pairs, indice_pair_num, outids.shape[0], subm=False, algo=algo)
    output_tensor = NPSparseConvTensor(out_features, outids, out_spatial_shape, batch_size)
    return output_tensor


def subm_conv3d(input: NPSparseConvTensor, weight: np.ndarray, in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, use_hash=False, algo=0):
    return sparse_convolution(input, weight, 3, in_channels, out_channels, kernel_size=kernel_size,
                       stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, subm=True,
                       indice_key=indice_key, use_hash=use_hash, algo=algo)


def sparse_conv3d(input: NPSparseConvTensor, weight: np.ndarray, in_channels, out_channels, kernel_size, 
                  stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, use_hash=False, algo=0):
    return sparse_convolution(input, weight, 3, in_channels, out_channels,
                       kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                       bias=bias, subm=False, indice_key=indice_key, use_hash=use_hash, algo=algo)


if __name__ == '__main__':
    np.random.seed(0)
    input_channels = 16
    out_channels = 32

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
    tl_outids, tl_indice_pairs, tl_indice_pair_num = get_indice_pairs(sp_indices, N, spatial_shape, [T, R, S], [stride_d, stride_h, stride_w], [pad_d, pad_h, pad_w], [dila_d, dila_h, dila_w], [0, 0, 0], True, False, -1)
    tl_w = np.ascontiguousarray(np.transpose(w, (2, 3, 4, 0, 1)))
    tl_x = NPSparseConvTensor(values, sp_indices, spatial_shape, N)
    tl_subm_y = subm_conv3d(tl_x, tl_w, C, K, [T, R, S], stride=[stride_d, stride_h, stride_w], 
                       padding=[pad_d, pad_h, pad_w], dilation=[dila_d, dila_h, dila_w], bias=False)
    tl_sparse_y = sparse_conv3d(tl_x, tl_w, C, K, [T, R, S], stride=[stride_d, stride_h, stride_w], 
                       padding=[pad_d, pad_h, pad_w], dilation=[dila_d, dila_h, dila_w], bias=False)
    check = False
    if check:
        '''
        only support spconv 1.x check, the result of spconv 1.x result is not same as torch?
        pip install spconv==1.2.1
        pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
        '''
        import torch
        import spconv
        
        # get_indice_pairs check
        sp_outids, sp_indice_pairs, sp_indice_pair_num = spconv.ops.get_indice_pairs(
                            torch.Tensor(sp_indices).int(),
                            N,
                            spatial_shape,
                            [T, R, S],
                            [stride_d, stride_h, stride_w],
                            [pad_d, pad_h, pad_w],
                            [dila_d, dila_h, dila_w],
                            [0, 0, 0],
                            True,
                            False,
                            None,
                            False)
        assert np.allclose(sp_outids.numpy(), tl_outids, atol=1e-4, rtol=1e-4)
        # assert np.allclose(sp_indice_pairs.numpy(), tl_indice_pairs, atol=1e-4, rtol=1e-4)
        assert np.allclose(sp_indice_pair_num.numpy(), tl_indice_pair_num, atol=1e-4, rtol=1e-4)

        # sparse tensor check
        sp_x = spconv.SparseConvTensor(torch.Tensor(values).float(), torch.Tensor(sp_indices).int(), spatial_shape, N) # [N, C, D, H, W]
        coo_x = torch.sparse_coo_tensor(indices, values, (D, H, W, C)) # [D, H, W, C]
        assert torch.allclose(sp_x.dense()[0], coo_x.to_dense().permute(3, 0, 1, 2), atol=1e-4, rtol=1e-4)

        # SubMConv3d check
        sp_subm_conv3d = spconv.SubMConv3d(C, K, [T, R, S], stride=[stride_d, stride_h, stride_w], padding=[pad_d, pad_h, pad_w], dilation=[dila_d, dila_h, dila_w], bias=False)
        sp_subm_conv3d.weight.data = torch.Tensor(tl_w).float()
        sp_subm_y = sp_subm_conv3d(sp_x)
        assert np.allclose(sp_subm_y.indices.detach().numpy(), tl_subm_y.indices, atol=1e-4, rtol=1e-4)
        assert np.allclose(sp_subm_y.spatial_shape, tl_subm_y.spatial_shape, atol=1e-4, rtol=1e-4)
        assert np.allclose(sp_subm_y.features.detach().numpy(), tl_subm_y.features, atol=1e-4, rtol=1e-4)

        # Conv3d check
        sp_sparseconv3d = spconv.SparseConv3d(C, K, [T, R, S], stride=[stride_d, stride_h, stride_w], padding=[pad_d, pad_h, pad_w], dilation=[dila_d, dila_h, dila_w], bias=False)
        sp_sparseconv3d.weight.data = torch.Tensor(tl_w).float()
        sp_sparse_y = sp_sparseconv3d(sp_x)
        assert np.allclose(sp_sparse_y.indices.detach().numpy(), tl_sparse_y.indices, atol=1e-4, rtol=1e-4)
        assert np.allclose(sp_sparse_y.indices.detach().numpy(), tl_sparse_y.indices, atol=1e-4, rtol=1e-4)
        assert np.allclose(sp_sparse_y.spatial_shape, tl_sparse_y.spatial_shape, atol=1e-4, rtol=1e-4)
        if not np.allclose(sp_sparse_y.features.detach().numpy(), tl_sparse_y.features, atol=1e-4, rtol=1e-4):
            print(f'spconv.SparseConv3d and sparse_conv3d differ')
            print(sp_sparse_y.features.detach().numpy())
            print(tl_sparse_y.features)
        assert np.allclose(sp_sparse_y.features.detach().numpy(), tl_sparse_y.features, atol=1e-4, rtol=1e-4)

        # torch.nn.Conv3d check
        # i
        conv3d = torch.nn.Conv3d(C, K, (T, R, S), stride=(stride_d, stride_h, stride_w), padding=(pad_d, pad_h, pad_w), dilation=(dila_d, dila_h, dila_w), bias=False).float()
        conv3d.weight.data = torch.Tensor(w)
        torch_features = conv3d(coo_x.to_dense().permute(3, 0, 1, 2).reshape(1, C, D, H, W))
        if not torch.allclose(sp_sparse_y.dense(), torch_features, atol=1e-4, rtol=1e-4):
            assert torch.allclose(sp_sparseconv3d.weight.data, conv3d.weight.data.permute(2, 3, 4, 0, 1), atol=1e-4, rtol=1e-4)
            assert torch.allclose(sp_x.dense(), coo_x.to_dense().permute(3, 0, 1, 2).reshape(1, C, D, H, W), atol=1e-4, rtol=1e-4)
            print(sp_sparse_y.dense())
            print(torch_features)




