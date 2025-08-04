### Environment Installation 
```
conda create -n dfuenv python=3.11.9
```


### Data Prepping
- Create Below
```
path-to: /home/chiawei/Documents/work/dfu

repo at <path-to>/DFUTissueSegNet

# Data
# mkdir <path-to>/DFUTissueSegNet_metadata
# - checkpoints
# - predictions
# - plots

in the above folder 
# mkdir -p dataset_MiT_v3+aug-added/{PNGImages,SegmentationClass,test_images,test_labels}

```
- Run `/dataset/createdata.py`
```
python createdata.py
```

### Model Training

- with cuda version 12.8
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

<details>


Package                  Version
------------------------ ------------
filelock                 3.13.1
fsspec                   2024.6.1
Jinja2                   3.1.4
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.3
numpy                    2.1.2
nvidia-cublas-cu12       12.8.3.14
nvidia-cuda-cupti-cu12   12.8.57
nvidia-cuda-nvrtc-cu12   12.8.61
nvidia-cuda-runtime-cu12 12.8.57
nvidia-cudnn-cu12        9.7.1.26
nvidia-cufft-cu12        11.3.3.41
nvidia-cufile-cu12       1.13.0.11
nvidia-curand-cu12       10.3.9.55
nvidia-cusolver-cu12     11.7.2.55
nvidia-cusparse-cu12     12.5.7.53
nvidia-cusparselt-cu12   0.6.3
nvidia-nccl-cu12         2.26.2
nvidia-nvjitlink-cu12    12.8.61
nvidia-nvtx-cu12         12.8.55
pillow                   11.0.0
pip                      25.1
setuptools               78.1.1
sympy                    1.13.3
torch                    2.7.1+cu128
torchaudio               2.7.1+cu128
torchvision              0.22.1+cu128
triton                   3.3.1
typing_extensions        4.12.2
wheel                    0.45.1

</details>

### Label
```
0: Background  
1: Granulation tissue  
2: Callus  
3: Fibrin  
4: Necrotic tissue  
5: Eschar  
6: Neodermis  
7: Tendon  
8: Dressing
```

