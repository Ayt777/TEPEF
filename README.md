Turbulence effects induced $C_n^2$ probing and evolutionary framework (TEPEF)
======
Overview
-------
Atmospheric turbulence, prevalent in natural and engineering systems, presents significant challenges for probing 3D refractive index structure constant $C_n^2$ due to its inherent randomness, complex structures, and spatiotemporal coupling. Despite the complexities, the optical degradations induced by multi-directional turbulence in multi-view images create an opportunity to learn the hidden 3D $C_n^2$ through turbulence imaging. In this study, a turbulence effects induced $C_n^2$ probing and evolutionary framework, termed TEPEF, is proposed. The TEPEF is composed of two sub-networks: the turbulence imaging effects induced 3D $C_n^2$ probing network (TIEPN) and turbulence spatiotemporal effects induced 3D $C_n^2$ evolution network (TSEEN), achieving accurate atmospheric turbulence probing and evolution prediction. The accurate acquisition of 3D $C_n^2$ provides, for the first time, a precise approach for obtaining comprehensive statistical turbulence properties. Additionally, we have demonstrated the practical utility of TEPEF in astronomical site selection, achieving accurate estimations of atmospheric coherence length $r_0$ and $seeing$. 

System requirements
-------

### Hardware requirements
No non-standard hardware required.

### Software requirements
The code has been tested on the following system:
* Linux: Ubuntu 20.04

<details>
  <summary>Requirements</summary>

  - Linux 
  - Python 3.7+
  - PyTorch 1.8 or higher
  - CUDA 10.1 or higher

</details>

<details>
  <summary>Dependencies</summary>

  - argparse
  - numpy
  - opencv-python
  - python<=3.10.8
  - scikit-image
  - scikit-learn
  - torch
  - tqdm
  - timm

</details>

Please refer to **Installation Guide** for more detailed instructions.

Installation Guide
-------
* TIEPN

* TSEEN
The code for TSEEN is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL/tree/OpenSTL-Lightning). Please refer to [OpenSTL](https://github.com/chengtan9907/OpenSTL/tree/OpenSTL-Lightning) for installation.

Demo & Instructions for use
-------
* TIEPN

* TSEEN

A small dataset is provided with code. You can simply run a demo:
```python
cd TSEEN/examples
python train.py
```
The operation will perform training and testing, save the checkpoints and the prediction result. To evaluate the result and obtain MSE, MAE, RMSE, $R^2$ and Pearson correlation coefficient, you can run:
```python
python R2.py
```

Test results for reproduction
-------
You can download the test results through [Baidu Netdisk](https://pan.baidu.com/s/1V52Dbm9ie3lHXgRNmV-eFQ?pwd=pcsy) or [Google Drive]().

License
-------
This project is covered under the **Apache 2.0 License**.
