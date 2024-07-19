# National University of Singapore [(NUS)](https://nus.edu.sg/) School of Computing [(SOC)](https://www.comp.nus.edu.sg/) [Summer Workshop](https://sws.comp.nus.edu.sg/) Course Project 2024
## Group2 project for the Robotics/Deep Learning

---

## Team Member
+ **Deep Learning Track**
  + YANG RUNKANG @***ShanghaiTech University*** major in ***Computer Science***
  + WEN SIJIE @***BUPT*** major in ***Software Engineering***
+ **Robotics Track**
  + ZHANG CHENHAN @***SUST*** major in ***Robotics Engineering***
  + HE JUNHANG  @***XJTU*** major in ***Artificial Intelligence***
---
## Summer Workshop Schedule
+ **The first week**
  + Prepare for the baseline project
+ **The second week**
  + Prepare for the demo of our own advanced model
+ **The third week**
  + Prepare for the final presentation of our own advanced model
---
## Baseline project
### Introduction

In our baseline project, we worked as a team of four, with two members focusing on deep learning and two on robotics, to build a remotely piloted vehicle. The objective was to navigate a maze, locate pictures of cats, and accurately identify their breeds. This document outlines our approach to collecting sample images of various cat species and training a deep learning model to recognize them.

### Image Collection

We collected images of the following five species of cats from the Internet:

- Ragdolls
- Singapura cats
- Persian cats
- Sphynx cats
- Pallas cats

Our goal was to gather over 1,000 images, ensuring an equal number of pictures for each cat type. We divided the images into 85% for training and 15% for validation. To streamline the process, we utilized web scraping techniques with Python and Selenium.

### Selecting and Training Deep Learning Networks

#### Model Selection

For the cat identification task, we chose a two-step approach using YOLO and a Convolutional Neural Network (CNN) for transfer learning. This method allowed us to first locate and identify the bounding boxes around the cats in the images, and then classify the cat breeds within those boxes.

1. **YOLO (You Only Look Once):** We used YOLO to detect and draw bounding boxes around the cats in the images. YOLO is known for its speed and accuracy in object detection, making it an ideal choice for this task.

2. **Transfer Learning with CNN:** After detecting the cats, we used a pre-trained Inception network for transfer learning to classify the cat breeds within the bounding boxes. The process involved fine-tuning the Inception network, which is well-regarded for its performance in image classification tasks.

#### Transfer Learning Process

The transfer learning process involved the following steps:

1. **YOLO Detection:** We trained a YOLO model to accurately detect and draw bounding boxes around the cats in our dataset.
2. **Base Model Selection:** For classification, we used the Inception network, known for its efficiency and high accuracy in image classification tasks.
3. **Additional Layers:** We added custom classification layers to adapt the model for our cat identification task.
4. **Training:** We fine-tuned the Inception network using the detected cat images, ensuring it could accurately identify the five cat species.

#### Justification

The Inception network provided a balanced trade-off between complexity and performance. By leveraging transfer learning, we were able to achieve high accuracy without the need for extensive computational resources or training time. The added layers allowed us to tailor the model specifically to our dataset, improving its ability to distinguish between the cat species.

---
## Advanced model



## Our Poster
![SWS3009_02 Poster](https://github.com/Rankyer/NUS-SOC-SummerWorkshop-Project/blob/main/poster/SWS3009_02.png?raw=true)

