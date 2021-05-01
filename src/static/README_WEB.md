# AIAP Batch 7 Assignment 7 Food Classifier

## Contents
1. Overview
2. Folder Structure
3. Setup
4. Model Architecture and Training
5. Model Performance
6. End Product
7. CI/CD

## 1. Overview
This web app was created as part of the AIAP7 Assignment 7, where the end goal is to be able to classify SG food after uploading an image. A model built in Assignment 5 for object classification using a pre-trained network was used for the predictions. Currently the app is only able to predict the following 12 food types:
- chilli crab
- curry puff
- dim sum
- ice kacang
- kaya toast
- nasi ayam
- popiah
- roti prata
- sambal stingray
- satay
- tau huay
- wanton noodle


## 2. Folder Structure
```
assignment7
+-- env.sh
+-- README.md
+-- Dockerfile
+-- conda.yml
+-- CODE_OF_CONDUCT.md
+-- INSTRUCTIONS.md
+-- skaffold.yaml
+-- ci
+-- src
|   +-- __init__.py
|   +-- app.py
|   +-- inference.py
|   +-- static
|   |   +-- main.css
|   |   +-- main.js
|   +-- templates
|   |   +-- base.html
|   |   +-- index.html
+-- tests
    +-- __init__.py
    +-- test_inference.py
```


## 3. Setup
Create an environment using the dependencies found in the `conda.yml` file.
```
conda create -n <environment_name> -f conda.yml
```

To run the app locally, launch the `app.py` file.
```
python -m src.app
```

Once ready, go to a browser and go the the following URL.
```
https://localhost:8000
```

Upload your desired image and click the `Submit` button to get the prediction as well as the predicted probability.


## 4. Model Architecture and Training
The dataset provided for training had a total of 1224 images, each belonging to one of the 12 food types mentioned in Section 1 above. This was then split into train, validation and test sets with ratios of 0.6, 0.2 and 0.2 respectively.

MobileNetV2 was used as a base model without the top layers, with pre-trained ImageNet weights.
Dimensions of 224x224x3 were used for inputs. Layers used can be found below:
- MobileNetV2 (with input dims of 224x224x3)
- Global Average Pooling
- Dense layer (256 nodes, relu activation)
- Dense layer (128 nodes, relu activation)
- Dropout layer (0.2)
- Output layer (softmax activation)

Model was then compiled with:
- Optimiser: Adam with learning rate of 0.0001
- Loss function: Categorical Cross Entropy

Training used 20 epochs with early stopping (based on validation loss with patience=3) and a batch size of 32. However, early stopping was not activated during training (shown in the next section).

## 5. Model Performance
### Training results
At epoch 20 (no early stopping), these were the results:
- Train loss: 0.2925 
- Train accuracy: 0.9179 
- Val loss: 0.4135 
- Val accuracy: 0.8719

The diagrams below show the plots of train/val accuracy and loss across the epochs.

![Train/val acc/loss](/assignment7/readme/train_val_acc_loss.png)

### Test results
- Test loss: 0.3546
- Test accuracy: 0.9044

The confusion matrix and classification results of the test set can be found below:

![Confusion mat](/assignment7/readme/confusion_matrix.png)

                     precision    recall  f1-score   support
    
        chilli_crab       0.49      0.95      0.65        21
         curry_puff       1.00      0.86      0.92        21
            dim_sum       0.88      0.83      0.85        35
         ice_kacang       1.00      0.83      0.91        18
         kaya_toast       0.85      0.85      0.85        20
          nasi_ayam       0.77      0.59      0.67        17
             popiah       0.93      0.67      0.78        21
         roti_prata       0.95      1.00      0.98        21
    sambal_stingray       1.00      0.91      0.95        22
              satay       0.90      0.86      0.88        21
           tau_huay       0.92      0.86      0.89        14
      wanton_noodle       0.95      1.00      0.98        20

           accuracy                           0.85       251
          macro avg       0.89      0.85      0.86       251
       weighted avg       0.89      0.85      0.86       251


## 6. End Product
An app with a simple UI for the user to upload an image for prediction was created using Flask and a basic html template. This was then deployed via Docker.

To use the app, only 3 steps are required:
1. Download a food image (from one of the 12 classes) of choice
2. Go to http://ruihanlim.aiap.okdapp.tekong.aisingapore.net/ and upload/drop the downloaded image from step 1 into the box bounded by dotted lines
3. Click submit and wait (it'll only be a second) for your results!


## 7. CI/CD
Continuous Integration (CI) is a software development practice for developers to merge code changes in a central repository with proper version control. Each time a change is made, a pipeline builds and tests the new code before actually merging it onto the shared repository. In doing so, the quality of the code can be maintained.

Continuous Deployment (CD) is the next step from CI where it ensures any code that has passed CI can be automatically released to production (after testing and staging) for users at any time.

Together, this will help developers to detect errors early and rectify any issues in the code before it's even merged into the repository. This way, the central repo can be considered as a "gold-standard" that has a low chance of having bugs when the product is automatically pushed to deployment in CD.

![CI CD 1](/assignment7/readme/ci_cd_1.png)

Note: Continuous Deployment is an extension of Continuous Delivery where the main difference is that deployment to production is automated in Deployment while in Delivery it's manual

![CI CD 2](/assignment7/readme/ci_cd_2.png)
