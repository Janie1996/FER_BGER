# Learning Facial Expression and Body Gesture Visual Information for Video Emotion Recognition

PyTorch implementation for the paper:

- Title: Learning Facial Expression and Body Gesture Visual Information for Video Emotion Recognition

- Authors: Jie Wei, Guanyu Hu,  Xinyu Yang, Luu Anh Tuan, Yizhuo Dong

- Submitted to: EXPERT SYSTEMS WITH APPLICATIONS

- Abstract: Recent research has shown that facial expressions and body gestures are two significant implications in identifying human emotions. However, these studies mainly focus on contextual information of adjacent frames, and rarely explore the spatio-temporal relationships between distant or global frames. In this paper, we revisit the facial expression and body gesture emotion recognition problems, and propose to improve the performance of video emotion recognition by extracting the spatio-temporal features via further encoding temporal information. Specifically, for facial expression, we propose a super image-based spatio-temporal convolutional model (SISTCM) and a two-stream LSTM model to capture the local spatio-temporal features and learn global temporal cues of emotion changes. For body gestures, a novel representation method and an attention-based channel-wise convolutional model (ACCM) are introduced to learn key joints features and independent characteristics of each joint. Extensive experiments on five common datasets are carried out to prove the superiority of the proposed method, and the results proved learning two visual information leads to significant improvement over the existing sota methods.

## Getting Started

```git
git clone https://github.com/Janie1996/FER_BGER.git
```

## Requirements

You can create an anaconda environment with:

```
conda env create -f environment.yaml
conda activate MSRFG
```

## Usage

### 1. Facial Expression
![img1](img1.jpg)
The proposed facial expression-based approach for emotion recognition is introduced in this section. As shown in Fig.1, the method includes three parts: video pre-processing, spatio-temporal features extraction, and emotion recognition. Firstly, the original video is divided into a certain number of clips, and only face part in sequences are kept through the pre-processing module. Secondly, the frames of each clip are used as the input of SISTCM to obtain local spatio-temporal features and clip-level emotion representations. Finally, the local spatio-temporal features and the clip-level emotion representations are simultaneously sent to the two-stream LSTM model to learn the global temporal relationship of facial expressions. 

#### 1. Preparation
    
    a. Download dataset  (Take CK+ for example)
        'http://www.jeffcohn.net/Resources/'

    b. Use DBFace to detect and position face 
        'https://github.com/dlunion/DBFace
    
    c. Download model checkpoint from google-drive. Unzip it and put them under ./Facial/Data/

#### 2. Test
    Run SISTCM + TLSTM (proposed)

        'python Facial/src/eval_CK+.py'

#### 3. Train
    Run SISTCM + TLSTM (proposed)

        'python Facial/src/train_CK+.py'
    
    Run Resnet + TLSTM

        'python Facial/src/train_resnet_CK+.py'

### 2. Body Gesture Emotion Recognition
In this section, we introduce the proposed body gesture-based method for emotion recognition. The method consists of three steps: body joints marking, body gesture representation, and emotion recognition. Firstly, the position of each joint in each video frame is marked. Secondly, we use different methods to represent the changes of body joints. Finally, the body gesture representation is sent to the ACCM to further learn features and recognize emotions.

#### 1. Preparation
    
    a. Download dataset (Take Emily for example)
    
    b. Use Openpose or PoseNet to extract joints position
        'https://github.com/CMU-Perceptual-Computing-Lab/openpose'
        'https://github.com/Hzzone/pytorch-openpose'
        
        'https://github.com/rwightman/posenet-python'
        'https://github.com/rwightman/posenet-pytorch'

    c. Download model checkpoint from google-drive. Unzip it and put them under ./Gesture/Data/

#### 2. Test
    Run (proposed)

        

#### 3. Train
    1. Body Gesture Representation
        - Body Gesture Representation Without Timeline:   x_y_one_potion.py    
        - Body Gesture Representation With Timeline:      x_y_oneroot.py
    2. Experiments
         'python Gesture/src/x_y_cnn.py'


More details coming soon ...

If you have questions, feel free to contact weijie_xjtu@stu.xjtu.edu.cn

## Acknowledgements
