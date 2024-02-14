# 🐼 Panda-70M
This is the offical Github repository of Panda-70M.

**[Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers](https://snap-research.github.io/Panda-70M)**
</br>
Tsai-Shien Chen,
Aliaksandr Siarohin,
Willi Menapace,
Ekaterina Deyneka,
Hsiang-wei Chao,
Byung Eun Jeon,
Yuwei Fang,
Hsin-Ying Lee,
Jian Ren,
Ming-Hsuan Yang,
Sergey Tulyakov

<!-- [Arxiv Report](https://arxiv.org/abs/2307.04725) | [Project Page](https://snap-research.github.io/Panda-70M) -->
[![arXiv](https://img.shields.io/badge/arXiv-2312.00000-b31b1b.svg)](https://arxiv.org/abs/2312.00000)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://snap-research.github.io/Panda-70M)

## Introduction
Panda-70M is a large-scale dataset with 70M high-quality video-caption pairs.
This repository have three sections:
- [Dataset Dataloading](./dataset_dataloading) includes the csv files listing the data of Panda-70M and the code to download the dataset.
- [Splitting](./splitting) includes the code to split a long video into multiple semantics-consistent short clips.
- [Captioning](./captioning) includes the proposed video captioning model trained on Panda-70M.

## Dataset
### Collection Pipeline
<p align="center" width="100%">
<a target="_blank"><img src="assets/collection_pipeline.gif" style="width: 100%; min-width: 200px; display: block; margin: auto;"></a>
</p>

### Download
  | Split           | Download | # Source Videos | # Samples | Video Duration | Storage Space|
  |-----------------|----------|-----------------|-----------|----------------|--------------|
  | Training (full) | [link](https://drive.google.com/file/d/1DeODUcdJCEfnTjJywM-ObmrlVg-wsvwz/view?usp=sharing) (2.01 GB) | 3,779,763 | 70,723,513 | 167 khrs  | ~36 TB  |
  | Training (10M)  | [link](https://drive.google.com/file/d/1Lrsb65HTJ2hS7Iuy6iPCmjoc3abbEcAX/view?usp=sharing) (381 MB)  | 3,755,240 | 10,473,922 | 37.0 khrs | ~8.0 TB |
  | Training (2M)   | [link](https://drive.google.com/file/d/1jWTNGjb-hkKiPHXIbEA5CnFwjhA-Fq_Q/view?usp=sharing) (86.5 MB) | 800,000   | 2,400,000  | 7.56 khrs | ~1.6 TB |
  | Validation      | [link](https://drive.google.com/file/d/1cTCaC7oJ9ZMPSax6I4ZHvUT-lqxOktrX/view?usp=sharing) (803 KB)  | 2,000     | 6,000      | 18.5 hrs  | ~4.0 GB |
  | Testing         | [link](https://drive.google.com/file/d/1ee227tHEO-DT8AkX7y2q6-bfAtUL-yMI/view?usp=sharing) (803 KB)  | 2,000     | 6,000      | 18.5 hrs  | ~4.0 GB |

More details can be found in [Dataset Dataloading](./dataset_dataloading) section.
  
## Demonstration
### Video-Caption Pairs in Panda-70M
  <table class="center">
    <tr>
      <td width=33.3% style="border: none"><img src="./assets/aIPu1xGNbhc.49.gif"></td>
      <td width=33.3% style="border: none"><img src="./assets/AIyw1FO1aqs.57.gif"></td>
      <td width=33.3% style="border: none"><img src="./assets/Kb8ON0iCs38.97.gif"></td>
    </tr>
    <tr style="text-align: center;">
      <td width=33.3% style="border: none">A rhino and a lion are fighting in the dirt.</td>
      <td width=33.3% style="border: none">A person is holding a long haired dachshund in their arms.</td>
      <td width=33.3% style="border: none">A rocket launches into space on the launch pad.</td>
    </tr>
  </table>

  <table class="center">
    <tr>
      <td width=33.3% style="border: none"><img src="./assets/AvVDsFBc6bA.0.gif"></td>
      <td width=33.3% style="border: none"><img src="./assets/S-1NdEjjg7c.58.gif"></td>
      <td width=33.3% style="border: none"><img src="./assets/10Y6wIEuG00.62.gif"></td>
    </tr>
    <tr style="text-align: center;">
      <td width=33.3% style="border: none">A person is kneading dough and putting jam on it.</td>
      <td width=33.3% style="border: none">A little boy is playing with a basketball in the city.</td>
      <td width=33.3% style="border: none">A 3d rendering of a zoo with animals and a train.</td>
    </tr>
  </table>

  <table class="center">
    <tr>
      <td width=33.3% style="border: none"><img src="./assets/_uQs-YDb5VA.9.gif"></td>
      <td width=33.3% style="border: none"><img src="./assets/CgcadSRtAag.140.gif"></td>
      <td width=33.3% style="border: none"><img src="./assets/1NMpoAqzJfY.25.gif"></td>
    </tr>
    <tr style="text-align: center;">
      <td width=33.3% style="border: none">A person in blue gloves is connecting an electrical supply to an injector.</td>
      <td width=33.3% style="border: none">There is a beach with waves and rocks in the foreground, and a city skyline in the background.</td>
      <td width=33.3% style="border: none">It is a rally car driving on a dirt road in the countryside, with people watching from the side of the road.</td>
    </tr>
  </table>

<sup>**We will remove the video samples from our dataset / Github / project webpage as long as you need it. Please contact tsaishienchen at gmail dot com for the request.</sup>

Please check [here](https://snap-research.github.io/Panda-70M/more_samples) for more samples.

### Long Video Splitting and Captioning

https://github.com/tsaishien-chen/Panda-70M/assets/43384650/40fee411-6617-4285-9698-a9b5692aeab0

https://github.com/tsaishien-chen/Panda-70M/assets/43384650/c3c36d5c-d96c-4f9e-8677-56d49a659fa0
  
## License of Panda-70M

The video samples are collected from a publicly available dataset.
Users must follow [the related license](https://raw.githubusercontent.com/microsoft/XPretrain/main/hd-vila-100m/LICENSE) to use these video samples.

## Citation

If you find this project useful for your research, please cite our paper. :blush:

```bibtex
@article{chen2023panda70M,
    title   = {Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers},
    author  = {Chen, Tsai-Shien and Siarohin, Aliaksandr and Menapace, Willi and Deyneka, Ekaterina and Chao, Hsiang-wei and Jeon, Byung Eun and Fang, Yuwei and Lee, Hsin-Ying and Ren, Jian and Yang, Ming-Hsuan and Tulyakov, Sergey},
    journal = {arXiv preprint arXiv:2402.00000},
    year    = {2024},
}
```

## Contact Information
**Tsai-Shien Chen**: [tsaishienchen@gmail.com](mailto:tsaishienchen@gmail.com) 
