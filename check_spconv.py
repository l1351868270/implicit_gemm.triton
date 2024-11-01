import spconv
try: 
    spconv.__version__
    SPCONV_VERSION = 2
except:
    SPCONV_VERSION = 1

if SPCONV_VERSION == 2:
    import spconv.pytorch as spconv
else:
    import spconv

import torch

torch.manual_seed(0)
torch.backends.cudnn.enabled = False

if __name__ == '__main__': 
    # sparse tensor check
    N = 1; C = 16; D = 9; H = 9; W = 9; K = 16; T = 3; R = 3; S = 3; pad_d = 1; pad_h = 1; pad_w = 1; stride_d = 1; stride_h = 1; stride_w = 1; dila_d = 1; dila_h = 1; dila_w = 1
    
    num_ = D * H * W
    indices_d = torch.randint(0, D, (num_,), dtype=torch.int32).to(device='cuda')
    indices_h = torch.randint(0, H, (num_,), dtype=torch.int32).to(device='cuda')
    indices_w = torch.randint(0, W, (num_,), dtype=torch.int32).to(device='cuda')
    indices = torch.stack([indices_d, indices_h, indices_w], dim=0).to(device='cuda')
    indices = torch.unique(indices, dim=1).to(device='cuda')
    sp_indices = indices.T.contiguous().to(device='cuda')
    sp_indices = torch.cat([torch.zeros((sp_indices.shape[0], 1), dtype=sp_indices.dtype, device=sp_indices.device), sp_indices], dim=1).to(device='cuda', dtype=torch.int32)
    num_ = len(sp_indices)
    values = torch.randn(num_, C, dtype=torch.float32).to(dtype=torch.float, device='cuda')
    spatial_shape = [D, H, W]
    sp_x = spconv.SparseConvTensor(values, sp_indices.int(), spatial_shape, N) # [N, C, D, H, W]
    coo_x = torch.sparse_coo_tensor(indices, values, (D, H, W, C)) # [D, H, W, C]
    assert torch.allclose(sp_x.dense()[0], coo_x.to_dense().permute(3, 0, 1, 2), atol=1e-4, rtol=1e-4)

    w = torch.randn(K, C, T, R, S, dtype=torch.float32).to(device='cuda')
    if SPCONV_VERSION == 2:
        tl_w = w.permute(0, 2, 3, 4, 1).contiguous().to(dtype=torch.float, device='cuda')
    else:
        tl_w = w.permute(2, 3, 4, 0, 1).contiguous().to(dtype=torch.float, device='cuda')

    # Conv3d check
    sp_sparseconv3d = spconv.SparseConv3d(C, K, [T, R, S], stride=[stride_d, stride_h, stride_w], padding=[pad_d, pad_h, pad_w], dilation=[dila_d, dila_h, dila_w], bias=False, algo=0)
    sp_sparseconv3d.weight.data = tl_w
    sp_sparse_y = sp_sparseconv3d(sp_x)
    # print(sp_sparse_y.dense())
    conv3d = torch.nn.Conv3d(C, K, (T, R, S), stride=(stride_d, stride_h, stride_w), padding=(pad_d, pad_h, pad_w), dilation=(dila_d, dila_h, dila_w), bias=False).float()
    conv3d.weight.data = w
    torch_features = conv3d(coo_x.to_dense().permute(3, 0, 1, 2).reshape(1, C, D, H, W))
    # print(torch_features)
    if not torch.allclose(sp_sparse_y.dense(), torch_features, atol=1e-4, rtol=1e-4):
        assert torch.allclose(sp_sparseconv3d.weight.data, conv3d.weight.data.permute(2, 3, 4, 0, 1), atol=1e-4, rtol=1e-4)
        assert torch.allclose(sp_x.dense(), coo_x.to_dense().permute(3, 0, 1, 2).reshape(1, C, D, H, W), atol=1e-4, rtol=1e-4)

        print(sp_sparse_y.dense())
        print(torch_features)