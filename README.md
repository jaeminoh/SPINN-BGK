<h1 align='center'> SPINN-BGK: Separable PINNs for the Boltzmann-BGK Model </h1>

[Paper](https://epubs.siam.org/doi/full/10.1137/24M1668809), [arXiv](https://arxiv.org/abs/2403.06342) 

## Abstract

In this study, we present a new approach based on [Separable Physics-Informed Neural Networks (SPINNs)](https://github.com/stnamjef/SPINN), specifically designed to efficiently solve the BGK model. While the mesh-free nature of Physics-Informed Neural Networks (PINNs) offers significant advantages for handling high-dimensional partial differential equations (PDEs), applying quadrature rules for accurate integral evaluation in the BGK operator can compromise these benefits and increase computational costs.
To address this issue, we leverage the canonical polyadic decomposition structure of SPINNs and the linear nature of moment calculation to significantly reduce the computational expense associated with quadrature rules. However, the multi-scale nature of the particle density function poses challenges for precisely approximating macroscopic moments using neural networks.
To overcome this, we introduce SPINN-BGK, a specialized variant of SPINNs that fuses SPINNs with Gaussian functions and utilizes a relative loss approach. This modification enables SPINNs to decay as rapidly as Maxwellian distributions, enhancing the accuracy of macroscopic moment approximations. The relative loss design ensures that both large and small-scale features are effectively captured by the SPINNs.
The effectiveness of our approach is validated through six numerical experiments, including a complex 3D Riemann problem. These experiments demonstrate the potential of our method to efficiently and accurately tackle intricate challenges in computational physics, offering significant improvements in computational efficiency and solution accuracy.

## Visualization
### 1D3V
**Kn=0.01**

https://github.com/user-attachments/assets/48dcfe42-be6a-41a2-990e-24bd956eb8d0

**Kn=1**


https://github.com/user-attachments/assets/c56ac8d8-98a6-4b68-8dd5-d663af1d2051




### 2D3V
**Kn=0.01**

https://github.com/user-attachments/assets/d9bf2abd-2150-4854-ac7a-8fb2ceaa2c9e

**Kn=1**

https://github.com/user-attachments/assets/9622bb81-675f-4e25-bed3-0b5067725887

### 3D3V
**Kn=0.01**
![3d_Kn0 01](https://github.com/user-attachments/assets/856c1739-f96d-4622-b780-4437f4c92698)

**Kn=1**
![3d_Kn1 0](https://github.com/user-attachments/assets/7d7600b9-6133-407c-ab23-168dd7af4306)



## How to run

### Dependency

- Python version: 3.12
- Install [JAX](https://jax.readthedocs.io/en/latest/installation.html)
- Install additional dependencies: run `pip install -r requirements.txt`

Change the arguments, if necessary.

### Training
Run `python smooth_3d.py --Kn=0.01 --rank=256` for the (3+3+1) smooth problem.

### Error tables
(3+3+1) dimensional problems: run `python error_3d.py --problem="smooth" --Kn=1.0`


## Citation
If you find this repository useful in your research, please consider citing us!

```bibtex
@article{oh2025separable,
  title={Separable physics-informed neural networks for solving the bgk model of the boltzmann equation},
  author={Oh, Jaemin and Cho, Seung Yeon and Yun, Seok-Bae and Park, Eunbyung and Hong, Youngjoon},
  journal={SIAM Journal on Scientific Computing},
  volume={47},
  number={2},
  pages={C451--C474},
  year={2025},
  publisher={SIAM}
}
```
