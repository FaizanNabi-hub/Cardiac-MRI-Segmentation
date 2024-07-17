# Cardiac-MRI-Segmentation
This study addresses left ventricle segmentation in Cardiac MRI using Encoder-Decoder architectures like Spatial Attention and U-Net, refined via dataset augmentation and tuning on the Sunny Brook dataset. Initial Dice and Jaccard coefficients of 0.4307 and 0.2834 improved to 0.8036 and 0.70. High GPU could push Dice above 0.9 and Jaccard above 0.8

This research work presents a structured approach to address the critical concerns associated with the accurate segmentation of Cardiac MRI images which is crucial for precise diagnosis, particularly in the challenging task of delineating the left ventricle. This study presents a systematic methodology for left ventricle segmentation, utilizing different Encoder-Decoder architectures, including Spatial Attention Encoder Decoder Architecture, Encoder Decoder Architecture and U-Net. Our approach begins with developing different Encoder Decoder architectures on MRI data and progresses to refining these models through architectural adjustments, dataset augmenting and hyperparameter tuning, using the curated Sunny Brook Left Ventricle dataset. Ethical considerations guide the meticulous dataset curation process. Preliminary results demonstrated promising improvements, with a Dice coefficient 0.4307 and a Jaccard coefficient 0.2834. By implementing Data Augmentation, Preprocessing images and masks, and incorporating a spatial attention mechanism into the Modified Spatial Attention U-Net architecture, we achieved state-of-the-art results with a significant improvement, achieving Dice coefficient of 0.8036 and Jaccard coefficient 0.70 and reaching a higher validation accuracy. Our model holds the potential to achieve even higher performance, with Dice coefficient exceeding 0.9 and Jaccard coefficient above 0.8, leveraging GPU power.  The study's contributions lie in its systematic approach, ethical adherence, and potential implications for advancing cardiac image analysis.


3. Methodology
3.1 Proposed Methodology 
Within the proposed methodology, the focus lies on delving into the requirements and limitations inherent in the envisaged system. This entails a meticulous examination of various facets such as data origins, computational capacities, and technical specifications of U-Net, types of U-Net, and its Modified Architectures
Data Sources:
The primary data source for this project is the curated Sunnybrook Left Ventricle dataset [5], consisting of Cardiac MRI images and corresponding segmentation masks. This dataset serves as the foundation for training, validation, and testing the deep learning models for left ventricle segmentation. Ethical considerations guide the process of dataset curation to ensure the integrity and representativeness of the data. 

Data Augmentation:
To address the limitations of our initial dataset containing 805 image-mask pairs, we implemented a controlled data augmentation strategy with the following parameters: rotations up to 180 degrees and enabling both horizontal and vertical flips. Zoom range, width shift range, and height shift range were all set to 0.0. This strategy aimed to expand the dataset size while introducing controlled variations that enhance model generalizability. Through these controlled transformations, the dataset was augmented to 3500 image-mask pairs. This process not only increased the data quantity by approximately 334.2% (from 805 to 3500) but also enriched its variability, enabling the model to learn robust features and generalize better to unseen data.


Data Preprocessing:
To prepare the dataset for training, we implemented a preprocess function to standardize the images and masks. The function resizes to 256x256 pixels using the PIL library, ensuring uniformity. After resizing, we add a channel dimension to both images and masks, making them suitable for convolutional neural networks. Image pixel values are normalized to the [0, 1] range by dividing by 255, which helps the neural network process the images effectively. For the masks, a binarization step converts pixel values to either 0 or 1 based on a threshold of 128, ensuring they are in binary format for segmentation tasks. This preprocessing pipeline standardizes the data and enhances the model's ability to learn meaningful features.

Model Loss and Monitoring Function
To evaluate and monitor model performance in our segmentation tasks, we use two key metrics: the Dice Coefficient and the Jaccard Coefficient. Additionally, we implement the Dice Loss function to optimize model training.
1) Dice Coefficient: The Dice Coefficient measures the overlap between predicted and true masks, ranging from 0 to 1, with higher values indicating better segmentation accuracy.
(2*|X∩Y|)/(|X|+|Y| )    (1)
2) Jaccard Coefficient (IoU): The Jaccard Coefficient, or Intersection over Union (IoU), quantifies the overlap between predicted and true masks, with a value of 1 indicating perfect alignment.
J(A,B)=(| A ∩ B |)/(| A ∪ B |)  =  (| A ∩ B |)/█(|A|+|B|-| A ∩ B| )    (2)
3) Dice Loss: The Dice Loss function transforms the Dice Coefficient into a minimizable metric for model training, incentivizing the model to enhance segmentation accuracy.
Dice Loss=1-Dice Coefficient   (3)

These metrics and loss functions are crucial for assessing and improving the accuracy and reliability of our segmentation models.

Modified Attention U-Net Architecture

![image](https://github.com/user-attachments/assets/2f89bbe0-5116-494a-abc7-8b6baf840e0e)

The segmentation model employs a modified U-Net architecture with an integrated encoder phase featuring multiple convolutional blocks followed by attention modules to progressively extract high-level features from input images. The decoder phase utilizes deconvolutional blocks with attention gates to upsample feature maps and enhance segmentation accuracy. Each convolutional block includes batch normalization for stable training. The attention mechanism, implemented in SpatialAttention, selectively amplifies informative features and integrates global average pooling for accurate segmentation. The model is optimized with stochastic gradient descent (SGD) with momentum and a Dice loss function to maximize segmentation performance, monitored using metrics like Dice coefficient and Jaccard index. Overall, this architecture not only emphasizes robust feature extraction and reconstruction through attention mechanisms but also ensures efficient training and validation processes critical for achieving high segmentation performance across diverse datasets.

Modified Spatial Attention U-Net Architecture and Results
![image](https://github.com/user-attachments/assets/217aad0a-2183-42bb-b7a0-61c6d076406e)
![image](https://github.com/user-attachments/assets/36d882cc-5c44-4263-bfac-b1962ebc5af9)







