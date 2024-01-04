# SAMA Overview

PyTorch implementation of **Scaling and Masking: A New Paradigm of Data Sampling for Image and Video Quality Assessment**, which has been accepted by **AAAI-2024**.

![](method.pdf)

## Usage

The method is simple enough. 

- Download the pretrained submodels first (see in the folder `pretrained_model`);
- Run `demo_extract_first.py` to extract features from dual streams. 
- Run `demo_run_main.py` or `demo_run_interdataset.py` to get the intra/inter-dataset performance in KoNViD-1K, LIVE-VQC, and YouTube-UGC. 

We have provided the extracted features in KoNViD-1K, LIVE-VQC, and YouTube-UGC in the folder `./data/`. Any other dataset would be ok with the same procedure.


### Environment
Different environment may induce possible fluctuation of performance.

```
Python 3.8.10
PyTorch 1.7.0
```


### Citation
If you are interested in the work, or find the code helpful, please cite our work
```
@ARTICLE{sama,  
  author={Liu, Yongxu and Quan, yinghui and Xiao, guoyao and Li, Aobo and Wu, jinjian},  
   journal={AAAI},   
   title={Scaling and Masking: A New Paradigm of Data Sampling for Image and Video Quality Assessment},   
   year={2024},  
   volume={},  
   number={},  
   pages={},  
   doi={}
}
```

### Contact

Feel free to contact me via `yongxu.liu@xidian.edu.cn` if any question or bug.
