# Handwritten-digit-recognition
Handwritten Digit Recognition using MNIST dataset, trained with DL4J and served via Spring Boot API.

---

## Steps Completed So Far

### 1. Training the Model
- Implemented `DataLoadingAndTraining.java`:
  - Loads **MNIST training and test datasets**.
  - Builds a CNN with convolution, pooling, dense, and output layers.
  - Trains for 7 epochs on both **normal and inverted images**.
  - Evaluates accuracy on the MNIST test set.
  - Saves trained model → `digit-model-cnn-both.zip`.

### 2. Generating Sample MNIST Images
- Implemented `GetMnistPictures.java`:
  - Exports one **normal** and one **inverted** image for each digit (0–9).
  - Saves them as `mnist_digit_X.png` and `mnist_digit_X_inverted.png`.

### 3. Prediction
- Implemented `Prediction.java`:
  - Loads `digit-model-cnn-both.zip`.
  - Preprocesses input image to **MNIST-style 28x28 grayscale**.
  - Runs prediction on both **normal and inverted versions** of the image.
  - Chooses result with higher confidence.
  - Prints predicted digit + probabilities for all digits.
- ✅ Can be run **directly on local images** without any server or API.  
  Example:
  ```bash
  java -jar target/digit-recog-ml-0.0.1-SNAPSHOT-jar-with-dependencies.jar "image-path"
