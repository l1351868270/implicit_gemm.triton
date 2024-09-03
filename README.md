# download
```
git clone --recurse-submodules -j8 https://github.com/l1351868270/implicit_gemm.triton.git
cd implicit_gemm.triton
```

# environment
```
conda create -n triton python=3.11
conda activate triton
conda install cuda==12.4.0 -c nvidia
conda install cudnn=9.1.1.17
conda install cmake=3.26.4
pip install nvidia_cudnn_frontend==1.6.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install ninja=1.10.2

rm -rf $(python -c "import sys; import os; print(os.path.dirname(sys.executable)+'/ld')")
```

# implicit_gemm.triton
```
python triton_bench.py
```
# cutlass_gemm

```
cutlass version v3.5.1

mkdir build
make 

```
