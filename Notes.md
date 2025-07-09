# Notes

## Terminology

1. 1 Epoch: 1 forward and backward pass of all training samples 
2. batch_size: number of training samples in one forward and backward pass 
3. number of iterations = number of passes, each passusing [batch_size] number of samples 

eg: 100 samples, batch0size = 20 ---> 100/20 = 5 iterations for 1 epoch 

## Journal 
### Making dataset
- using glob instead of listdir to make a csv file with all images, and their lables 
- Images are usually 640 x 480 in size, resizing them to 224 x 224 (suitable for resNet50 and other models)
- using a normalization as per ImageNet standard
- 221 healthy leaf images 
- 3292 total unhealthy leaf images
- In order to make the training and testing balances, I need to make sure that the distrubutions of images in each class are equal

- DATA AUGMENTAION: Due to the very small data size, I have decided to expand the dataset:
    1. Random brighness: I decided to include this since colour and visibility are important to the model
    2. Random shift
    3. Random zoom(?)


Sample images:
When images not normalized (pixel size kept to original):
![image](https://github.com/user-attachments/assets/36773fa8-d8cc-4945-b743-781e71bd25fa)
When images are resized to 224 x 224:
![image](https://github.com/user-attachments/assets/7d6f051a-c7ea-4bd3-87c6-5cc2788f7467)

When images are normalized (pixel size kept to original):
![image](https://github.com/user-attachments/assets/3a28023a-9ad1-4e5f-9916-29ca70ccd2be)

### Models
1. DenseNet121
2. ResNet50
3. MobileNetV2
4. EfficientNetB0

### versions
- v1: Deals with classweight imbalance, adds transformations ot images for data augmentation
- v2: After plotting and checking all results, I was not satisfied with f-1 scores of the models, hence v2 introduces k-fold cross validation and unfreezing backbone after a few epochs (adding onto v1)
### To D0
- Complete multi-class training
- plotting for multi-class


![image](https://github.com/user-attachments/assets/728bc6d3-70f7-4f3a-8116-d39f7fc08365)
From: Prasad et al. https://www.nature.com/articles/s41598-024-59562-x#ref-CR12


