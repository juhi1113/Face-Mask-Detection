# Face Mask Detection

This project aims to build a convolutional neural network (CNN) model to detect whether a person in an image is wearing a face mask or not. The model is trained on a dataset containing images of people with and without face masks.

## Dataset

The dataset used in this project consists of two directories:

- `with_mask`: Contains images of people wearing face masks.
- `without_mask`: Contains images of people not wearing face masks.

Dataset Link:https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

## Dependencies

The following dependencies are required to run this project:

- Python 3.x
- NumPy
- Matplotlib
- OpenCV
- Keras
- TensorFlow
- Scikit-learn
- Pillow


## Usage

1. Clone this repository or download the source code.
2. Prepare your dataset by placing the images in the `with_mask` and `without_mask` directories.
3. Update the file paths in the code to point to your dataset directories.
4. Run the script `face_mask_detection.py`.
5. When prompted, enter the path of the image you want to classify.
6. The script will display the input image and print whether the person in the image is wearing a mask or not.

## Model Architecture

The CNN model used in this project has the following architecture:

The model consists of two convolutional layers followed by max-pooling layers, a flattening layer, and two dense layers with dropout regularization. The final layer has two output nodes, representing the two classes (with mask and without mask).

## Training

The model is trained using the `model.fit()` function from Keras. The training data is split into training and validation sets using the `train_test_split` function from Scikit-learn. The model is trained for 5 epochs with the Adam optimizer and sparse categorical cross-entropy loss function.

## Evaluation

After training, the model is evaluated on the test set using the `model.evaluate()` function. The script prints the test accuracy score.

Additionally, the training and validation loss and accuracy curves are plotted using Matplotlib for visual inspection.

## Prediction

To predict whether a person in a new image is wearing a mask or not, the script prompts the user to enter the path of the image. The image is preprocessed (resized and scaled) and then passed through the trained model using the `model.predict()` function. The predicted label (0 for without mask, 1 for with mask) is printed to the console.
