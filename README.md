# National University of Singapore [(NUS)](https://nus.edu.sg/) School of Computing [(SOC)](https://www.comp.nus.edu.sg/) [Summer Workshop](https://sws.comp.nus.edu.sg/) Course Project 2024
## Group2 project for the Robotics/Deep Learning

---

## Team Member
+ **Deep Learning Track**
  + YANG RUNKANG @***ShanghaiTech University*** major in ***Computer Science***
  + WEN SIJIE @***BUPT*** major in ***Software Engineering***
+ **Robotics Track**
  + ZHANG CHENHAN @***SUSTech*** major in ***Robotics Engineering***
  + HE JUNHANG  @***XJTU*** major in ***Artificial Intelligence***
---
## Summer Workshop Schedule
+ **The first week**
  + Prepare for the baseline project
    + #### ***For more content related to deep learning, please refer to [this GitHub repository](https://github.com/Rankyer/NUS-SOC-SummerWorkshop-DeepLearning-Labs), which contains the 7 lab contents of the deep learning section.***
+ **The second week**
  + Prepare for the demo of our own advanced model
+ **The third week**
  + Prepare for the final presentation of our own advanced model
---
## Notes
#### ***For the dataset we used, please refer to [this GitHub repository](https://github.com/Rankyer/NUS-SOC-SummerWorkshop-Project-Dataset), which contains the baseline dataset (five kinds of cats) as well as the dataset of our advanced model.***
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
## Advanced model: An Intelligent Pet Community Delivery System
> ***Our Story***: Lily has three pets: cat,dog, and bird. She plans to travel but worries about her pets' care😔. So, she asked us to create a smart cart. This cart can recognize each pet using image recognition and deliver their favorite fruits，for example:apples for the cat, watermelon for the dog, and bananas for the bird. The cart delivers fresh fruits daily, ensuring the pets are well-fed. While Lily is away, the cart operates reliably, feeding the pets on schedule. When Lily returns, she finds her pets healthy and happy, satisfied with the invention. The cart not only solved her problem but also became a helpful companion for her pets😊.

### Introduction

Building on our baseline project, our team of four (two members focusing on deep learning and two on robotics) developed an innovative shopping recommendation system for pets using machine learning and deep learning techniques. The system is designed for an intelligent pet community, where our custom-designed delivery vehicle operates similarly to a food delivery robot, providing an efficient and automated delivery service. 

### Deep Learning Approach

#### Data Collection

We collected datasets for three types of fruits: apples, bananas, and watermelons, as well as for three types of animals: birds, cats, and dogs. This data was used to train our model, building on the methodology from our baseline project.

#### Model Training

We used the Inceptionv3 model for transfer learning to accurately classify the images in our dataset. Our process involved:

1. **Data Preparation:** Collecting and preprocessing images of fruits and animals.
2. **Training:** Utilizing the Inceptionv3 network, fine-tuning it to recognize and classify the collected images accurately.
3. **Optimization:** Ensuring the model's performance was optimized for high accuracy in classification tasks.

#### Image Recognition

To simplify and enhance the accuracy of our image recognition process, we printed images of the fruits and animals on paper, highlighted with bold borders. Using Python's OpenCV library, we detected these borders and corners, enabling us to crop and classify the contents efficiently, eliminating the need for YOLO and improving precision.

### Robotics Approach

#### Design and Construction

We utilized 3D printing technology to design and construct the necessary shelves and cargo for our delivery system. Our design included a novel delivery and pickup mechanism that uses physical interactions to manage the delivery and pickup process, offering better control compared to a robotic arm.

#### Navigation System

For the delivery vehicle, we implemented a track-following and obstacle-avoidance system using black lines and tracking modules. This ensured smooth and stable navigation along a fixed path. Additionally, we incorporated color modules for precise positioning:

- **Blue Color Blocks:** Indicate pickup points where the vehicle will stop to load cargo.
- **Red Color Blocks:** Indicate delivery points where the vehicle will stop to unload cargo.

This system significantly improved the accuracy of the vehicle's positioning, facilitating the subsequent image recognition process by the deep learning model.

### Integration of Deep Learning and Robotics

#### Communication and Control

Instead of using MQTT for communication, we migrated our trained model to a Raspberry Pi and built a simple front-end interface using Flask. This allowed users to manually add delivery orders to the queue and manage the system effectively.

#### Intelligent Recommendation System

To enhance the intelligence of our delivery system, we collected shopping data for all animals in the pet community. The front-end interface allows users to view the preferences of different animals at specific times and make informed decisions on food selection. This data-driven approach improves the efficiency and personalization of the delivery service.

#### Delivery Workflow

1. **Order Placement:** Users can either manually select or use the intelligent recommendation system to choose food for each animal based on their preferences.
2. **Queue Management:** Orders are added to the delivery queue.
3. **Vehicle Operation:** The delivery vehicle starts its journey, first performing the pickup process when it encounters a blue color block:
   - If the current cargo is not the desired item, the vehicle continues moving.
   - If the current cargo matches the order, the vehicle picks it up using the front-mounted mechanism, then reverses back to the main path.
4. **Delivery Process:** The vehicle proceeds to the delivery point, indicated by a red color block, and unloads the cargo.
5. **Completion:** The queue's first element is cleared, and the vehicle begins the next delivery cycle.

Our advanced model integrates deep learning and robotics to create an intelligent delivery system for a pet community. By combining Inceptionv3 for image recognition with a carefully designed robotic delivery mechanism, we have developed a reliable and efficient system. The front-end interface and intelligent recommendation system further enhance the user experience, making our pet community delivery system both innovative and practical.





## Our Poster
![SWS3009_02 Poster](https://github.com/Rankyer/NUS-SOC-SummerWorkshop-Project/blob/main/poster/SWS3009_02.png?raw=true)

