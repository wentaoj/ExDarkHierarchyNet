# ExDarkHierarchyNet: Enhancing Low-Light Image Classification with CNN Architectures

## Project Information
#### Author: Yuesong Huang†, Wentao Jiang†
#### Release Date: Dec.18, 2023
_Last Edit: Jan.26, 2024_

## Project Overview
**"ExDarkHierarchyNet"** is a computer vision project designed for enhancing image classification in extremely low-light conditions. It innovatively employs a hierarchical multi-label classification approach, achieving high accuracy in environments challenging even for human vision. This advancement holds significant potential for real-world applications such as enhancing Autonomous Driving Systems navigation at night and aiding in wildlife monitoring with critical needs in low-light image processing.

### Project Report
- **Project Presentation Slides** for ExDarkHierarchyNet: [`ExDarkHierarchyNet-Slides.pdf`](./ExDarkHierarchyNet-Slides.pdf)
- **Project Report Paper** for ExDarkHierarchyNet: [`ExDarkHierarchyNet-Report.pdf`](./ExDarkHierarchyNet-Report.pdf)

### Keywords:
Computer Vision, Deep Learning, Image Classification, Convolutional Neural Network, ExDark Dataset

## Dataset Overview
### Exclusively Dark (ExDark) Image Dataset (2018)
**Source:** [Exclusively Dark (ExDark) Image Dataset (Official Site)](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master)
The Exclusively Dark (ExDark) Image Dataset provides a diverse collection of low-light images that are essential for training and testing our models. The dataset features:
- **7,363 Low-Light Images**: Ranging from very low-light environments to twilight across 10 different conditions.
- **12 Object Classes**: Each image in the dataset comes with its own separate annotation, including local object bounding boxes and image class labels. To effectively utilize this dataset, we developed a unique [`ExDark_annotator` tools](./data/ExDark_annotator.py). This tool processes each image's annotation, performs calculations like average RGB values, and then merges the annotations to align with our classification pipeline.
- **Download Links**
  - Access the [ExDark Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset) / Official Dataset Download Link: [Google Drive (1.5Gb)](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view?usp=sharing)
  - and [ExDark Groundtruth](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Groundtruth) / Official Groundtruth Download Link: [Google Drive (4.2Mb)](https://drive.google.com/file/d/1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i/view?usp=sharing)


## Features, Models, and Results
For a more comprehensive view, please continue reading at [`ExDarkHierarchyNet.ipynb`](./ExDarkHierarchyNet.ipynb) and [`ExDarkHierarchyNet-Report.pdf`](./ExDarkHierarchyNet-Report.pdf).

- **Annotations**: Standard annotations `./data/image_annotations.csv` and merged annotations for multi-label training `./data/image_annotations_merged.csv`. *For a snapshot of these files, see [Annotation Data Overview:](#annotation-data-overview) in Appendix.*

- **Classification Strategy**: Employs a multi-stage and multi-label classification approach, encompassing binary, ternary, and N-ary classifications across 12 classes.

- **Architectures Framework**: Employs 3-layer CNN, ResNet50, and VGG19 models  throughout various classification stages, detailed in [Section 3.7: Models](./ExDarkHierarchyNet-Report.pdf).

- **Enhanced Image Processing**: Tailored for the ExDark Dataset using various transformation techniques to enhance classification in extremely dark environments. See examples in Jupyter notebook [`visualization_demo.ipynb`](./preprocessing/visualization_demo.ipynb) and [Section 3: Methods](./ExDarkHierarchyNet-Report.pdf).

- **Methodological Approach**: Involves hyper-parameter tuning, data augmentation, custom image preprocessing, and the integration and training of ResNet50 and VGG19 backbones, as elaborated in [Section 4: Experiments](./ExDarkHierarchyNet-Report.pdf).

- **Performance Metrics**: Up to 93.5818% in Binary Classification, 92.3977% in Ternary Classification, and 60.1494% in N-ary Multilabel Classification (12 classes) under extermely low-light environments.

- **Model Accessibility**: The trained models in `.pth` are available on [Google Drive](https://drive.google.com/drive/folders/1_nCNsZ-VjuJr1jbh5UTjZEzf-6pbBu4v?usp=sharing). After downloading, please place the models under `./models/` directory.


## Development Environment
- **Python Version**: Python 3.10.12 [GCC 11.4.0] on linux (Ubuntu 22.04.3 LTS / Google Colab)
- **Package Management**: pip 23.1.2
- **CUDA Compatibility**: CUDA 12.2 or above for GPU acceleration.
- **Libraries**: PyTorch, OpenCV(cv2), Scikit-Learn/Image, etc. See [`requirements.txt`](./requirements.txt) for details.
- **Dataset Location**: The ExDark Dataset and its Groundtruth are stored under [`./data/`](./data/) directory at `./data/ExDark/` for Dataset and `./data/ExDark_Annno/` for Groundtruth.

## Installations & Usage
1. Clone this repository to your local machine.
2. Check your current Python version with `python --version`. If not installed, download it from [Python](https://www.python.org/downloads/). For GPU acceleration and support, using High-end NVIDIA Graphics Cards, or Google Colab Pro+ subscription is recommended.
3. Install the required dependencies by running `pip install -r requirements.txt` in the project directory.
4. Download the ExDark Dataset and Groundtruth, placing them under the [`./data/`](./data/) directory.
   - ExDark Dataset: [Google Drive (1.5Gb)](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view?usp=sharing)
   - ExDark Groundtruth: [Google Drive (4.2Mb)](https://drive.google.com/file/d/1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i/view?usp=sharing)
5. For detailed project instructions and examples, refer to the included Jupyter notebook [`ExDarkHierarchyNet.ipynb`](./ExDarkHierarchyNet.ipynb).
6. Ready-to-Use Models: Trained models in .pth are available in [Google Drive](https://drive.google.com/drive/folders/1_nCNsZ-VjuJr1jbh5UTjZEzf-6pbBu4v?usp=sharing) for use.

## Copyright Info
- **Dataset Description**: [Exclusively Dark (ExDark) Image Dataset (Official Site)](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset)
- **Dataset License**: [BSD-3-Clause license](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/LICENSE)
- **Models Developed and Maintained by**: 
  - &copy; 2024 Wentao Jiang†, Yuesong Huang†.
- **License**: 
  - This project is licensed under the [BSD-3-Clause license](./LICENSE). Full license text is available in the repository.
- **Acknowledgments**: 
  - Our gratitude goes to to all the developers and maintainers of the essential resources used in this project, including the ExDark dataset, Python libraries, and the CNN architectures, particularly ResNet50 and VGG19 backbones.
  - Special thanks to authors of [ICLR2024 Master template](https://github.com/ICLR/Master-Template) for their contribution to the project report.

## Appendix
### Annotation Data Overview:
#### ExDark_Annotations.csv:
The ExDark Dataset comes with annotations for each image in the dataset. The annotations are stored in [`./data/ExDark_Annotations.csv (1.26Mb)`](./data/ExDark_Annotations.csv).
| image          | class   | x   | y   | w   | h   | fpath                     |
| -------------- | ------- | --- | --- | --- | --- | ------------------------- |
| 2015_00001.png | Bicycle | 204 | 28  | 271 | 193 | ./data/ExDark/Bicycle     |
| 2015_00002.png | Bicycle | 136 | 190 | 79  | 109 | ./data/ExDark/Bicycle     |
| 2015_00002.png | Bicycle | 219 | 172 | 63  | 131 | ./data/ExDark/Bicycle     |
| ...            | ...     | ... | ... | ... | ... | ...                       |

*Total Columns: 7*
*Total Rows: 23,710*

#### image_annotations_merged.csv:
To streamline the dataset annotations, we merged and analyzed the data associated with each image. The merged annotations are stored in [`./data/image_annotations_merged.csv(436Kb)`](./data/image_annotations_merged.csv).
| image          | class                         | class_index | label | fpath                     |
| -------------- | ----------------------------- | ----------- | ----- | ------------------------- |
| 2015_00001.png | ['Bicycle']                   | [0]         | 0     | ./data/ExDark/Bicycle     |
| 2015_00002.png | ['Bicycle', 'Car']            | [0, 4]      | 0     | ./data/ExDark/Bicycle     |
| 2015_00003.png | ['Bicycle', 'Bus', 'Chair']   | [0, 3, 6]   | 0     | ./data/ExDark/Bicycle     |
| ...            | ...                           | ...         | ...   | ...                       |

*Total Columns: 5*
*Total Rows: 7,362*

### Project Directory Structure:
```bash
.
├── Bib-ICLR2024
│   ├── fancyhdr.sty
│   ├── figure+object
│   │   └── *.png
│   ├── iclr2024_conference.bib
│   ├── iclr2024_conference.bst
│   ├── iclr2024_conference.sty
│   ├── iclr2024_conference.tex
│   ├── math_commands.tex
│   └── natbib.sty
├── ExDarkHierarchyNet-Report.pdf
├── ExDarkHierarchyNet-Slides.pdf
├── ExDarkHierarchyNet.ipynb
├── LICENSE
├── README.md
├── data
│   ├── ExDark_annotator.py
│   ├── image_annotations.csv
│   └── image_annotations_merged.csv
├── models
│   ├── README.md
│   └── model_placeholder # Place the `.pth' file here
├── preprocessing
│   ├── ExDark_imgEnhancer.py
│   ├── enhanced_images
│   │   └── *.png
│   └── visualization_demo.ipynb
└── requirements.txt

6 directories, 47 files
# Note: The structures in this visualization is modified for brevity.
```

&copy; 2024. Wentao Jiang