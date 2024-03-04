
# Palette: Imaging Dynamics Beneath Turbid Media via Parallelized Single-Photon Detection

[Paper](https://doi.org/10.1002/advs.202201885)

## Brief

Noninvasive optical imaging through dynamic scattering media has numerous important biomedical applications but still remains a challenging task. While standard diffuse imaging methods measure optical absorption or fluorescent emission, it is also well-established that the temporal correlation of scattered coherent light diffuses through tissue much like optical intensity. Few works to date, however, have aimed to experimentally measure and process such temporal correlation data to demonstrate deep-tissue video reconstruction of decorrelation dynamics. In this work, a single-photon avalanche diode array camera is utilized to simultaneously monitor the temporal dynamics of speckle fluctuations at the single-photon level from 12 different phantom tissue surface locations delivered via a customized fiber bundle array. Then a deep neural network is applied to convert the acquired single-photon measurements into video of scattering dynamics beneath rapidly decorrelating tissue phantoms. The ability to reconstruct images of transient (0.1–0.4 s) dynamic events occurring up to 8 mm beneath a decorrelating tissue phantom with millimeter-scale resolution is demonstrated, and it is highlighted how the model can flexibly extend to monitor flow speed within buried phantom vessels.


## Status

### Code
- [x] Diffusion Model Pipeline
- [x] Train/Test Process
- [x] Save/Load Training State
- [x] Logger/Tensorboard
- [x] Multiple GPU Training (DDP)
- [x] EMA
- [x] Metrics (now for FID, IS)
- [x] Dataset (now for inpainting, uncropping, colorization)
- [x] Google colab script 🌟(now for inpainting)

### Task

I try to finish following tasks in order:
- [x] Inpainting on [CelebaHQ](https://drive.google.com/drive/folders/1CjZAajyf-jIknskoTQ4CGvVkAigkhNWA?usp=sharing)🚀 ([Google Colab](https://colab.research.google.com/drive/1wfcd6QKkN2AqZDGFKZLyGKAoI5xcXUgO#scrollTo=8VFpuekybeQK))
- [x] Inpainting on [Places2 with 128×128 centering mask](https://drive.google.com/drive/folders/1fLyFtrStfEtyrqwI0N_Xb_3idsf0gz0M?usp=sharing)🚀

The follow-up experiment is uncertain, due to lack of time and GPU resources:

- [ ] Uncropping on Places2
- [ ] Colorization on ImageNet val set 

## Results

The DDPM model requires significant computational resources, and we have only built a few example models to validate the ideas in this paper.

### Visuals

#### Celeba-HQ

Results with 200 epochs and 930K iterations, and the first 100 samples in [centering mask](https://drive.google.com/drive/folders/10zyHZtYV5vCht2MGNCF8WzpZJT2ae2RS?usp=sharing) and [irregular mask](https://drive.google.com/drive/folders/1vmSI-R9J2yQZY1cVkSSZlTYil2DprzvY?usp=sharing). 

| ![Process_02323](img//fig1.jpg) |    ![Process_02323](misc//image//Process_26190.jpg)  |
| ------------------------------------------------ | ---- |

#### Places2 with 128×128 centering mask

Results with 16 epochs and 660K iterations, and the several **picked** samples in [centering mask](https://drive.google.com/drive/folders/1XusKO0_M6GUfPG-FOlID0Xcp0SiexKNe?usp=sharing).

| ![Mask_Places365_test_00209019.jpg](misc//image//Mask_Places365_test_00209019.jpg) | ![Mask_Places365_test_00143399.jpg](misc//image//Mask_Places365_test_00143399.jpg) | ![Mask_Places365_test_00263905.jpg](misc//image//Mask_Places365_test_00263905.jpg) |  ![Mask_Places365_test_00144085.jpg](misc//image//Mask_Places365_test_00144085.jpg)    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| ![Out_Places365_test_00209019](misc//image//Out_Places365_test_00209019.jpg) | ![Out_Places365_test_00143399.jpg](misc//image//Out_Places365_test_00143399.jpg) | ![Out_Places365_test_00263905.jpg](misc//image//Out_Places365_test_00263905.jpg) | ![Out_Places365_test_00144085.jpg](misc//image//Out_Places365_test_00144085.jpg)    |

#### Uncropping on Places2

Results with 8 epochs and 330K iterations, and the several  **picked** samples in [uncropping](https://drive.google.com/drive/folders/1tC3B8ayaadhXAJrOCTrw15R8t84REPWJ?usp=sharing).
| ![Process_Places365_test_00309553](misc//image//Process_Places365_test_00309553.jpg) |    ![Process_Places365_test_00042384](misc//image//Process_Places365_test_00042384.jpg)  |
| ------------------------------------------------ | ---- |

Noninvasive optical imaging through dynamic scattering media has numerous important biomedical applications but still remains a challenging task. While standard diffuse imaging methods measure optical absorption or fluorescent emission, it is also well-established that the temporal correlation of scattered coherent light diffuses through tissue much like optical intensity. Few works to date, however, have aimed to experimentally measure and process such temporal correlation data to demonstrate deep-tissue video reconstruction of decorrelation dynamics. In this work, a single-photon avalanche diode array camera is utilized to simultaneously monitor the temporal dynamics of speckle fluctuations at the single-photon level from 12 different phantom tissue surface locations delivered via a customized fiber bundle array. Then a deep neural network is applied to convert the acquired single-photon measurements into video of scattering dynamics beneath rapidly decorrelating tissue phantoms. The ability to reconstruct images of transient (0.1–0.4 s) dynamic events occurring up to 8 mm beneath a decorrelating tissue phantom with millimeter-scale resolution is demonstrated, and it is highlighted how the model can flexibly extend to monitor flow speed within buried phantom vessels.
