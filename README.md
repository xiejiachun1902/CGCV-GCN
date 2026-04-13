# Cognition-guided Complex-valued Graph Convolutional Network for Gait Emotion Recognition [TAFFC 2026]

<i>Jiachun Xie, Binyuan Huang, Faliang Huang*, Jiayi Yao, Yuting Zhang, Guangqing Long, Demin Wu, and Shuochun Chen</i>

Official implementation of the CGCV-GCN model. [[Paper Page]](https://ieeexplore.ieee.org/document/11477988)




## Introduction
Gait emotion recognition has played a significant role in the development of intelligent systems. Most existing methods primarily focus on capturing the spatial-local relationships of gait while paying insufficient attention to the temporal features of gait, despite the fact that temporal information contains critical cues for emotion recognition. To address the issue, in this article, a novel model, termed CGCV-GCN, is proposed for gait emotion recognition, which targets at improving the GCN network by integrating complex-valued representation and cognitive hierarchy. Specifically, a Complex-valued Spatial-Temporal Graph Convolutional Network (CST-GCN) is first devised for learning gait emotion representation in a magnitude-phase complex space, which leverages phase information to improve gait emotion features. Furthermore, a cognition-inspired fusion mechanism (CGFM) is proposed to adaptively integrate cross-representation-space and cross-modal features, enabling a multi-level local-global gait feature learning process. Experimental results on public datasets demonstrate that CGCV-GCN significantly improves emotion recognition accuracy over state-of-the-art methods. The code is available at:

## Running

1. Clone Repository
```
git clone https://github.com/xiejiachun1902/CGCV-GCN.git
```

2. Environment Setup and Dependency Installation
```
conda create --name CGCV-GCN python=3.9.0
conda activate CGCV-GCN
conda install pip
pip install -r requirements.txt
```

3. Dataset Preparation and Configuration

***Datasets***: The dataset utilized in this work is available for [download](https://drive.google.com/drive/folders/1wohc4sVNzOsDMweFVK-0iCdico_UmZKR?usp=sharing) via OneDrive

***Configuration***：Update the dataset path in '.../CGCV_GCN/config/EGait_journal/train_CGCV_GCN.yaml' to match your local directory. You may also customize hyperparameters, including batch size and the number of training epochs, as needed.

4. Training
```
python main_CGCV_GCN.py --config ./config/EGait_journal/train_CGCV_GCN.yaml
```

## Citation

If this research proves beneficial to your work or project, please consider citing the following publication:

**Bibtex Citation**
````
@ARTICLE{xie2026CGCVGCN,
  author={Xie, Jiachun and Huang, Binyuan and Huang, Faliang and Yao, Jiayi and Zhang, Yuting and Long, Guangqing and Wu, Demin and Chen, Shuochun},
  journal={IEEE Transactions on Affective Computing}, 
  title={Cognition-guided Complex-valued Graph Convolutional Network for Gait Emotion Recognition}, 
  year={2026},
  pages={1-13}
  }
````