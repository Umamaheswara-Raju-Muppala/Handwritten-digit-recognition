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


---

##  API Development (Spring Boot)

### 4.1 Project Setup
- Created a new **Spring Boot project** (`digit-recog-api`).
- Added dependencies:
  - `spring-boot-starter-web`
  - `deeplearning4j-core` (for loading trained model and prediction)
  - `nd4j-native-platform` (CPU backend for DL4J)
  - `slf4j-simple` (logging)
- Copied trained model (`digit-model-cnn-both.zip`) into project resources.

### 4.2 Service Layer
- Implemented `PredictionService.java`:
  - Loads trained CNN model at application startup.
  - Accepts uploaded image (`MultipartFile`).
  - Converts to **MNIST-style grayscale 28x28** format.
  - Runs prediction on both **normal and inverted images**.
  - Returns the digit with **highest confidence**.

### 4.3 Controller
- Implemented `PredictController.java`:
  - Exposes `POST /predict` endpoint.
  - Accepts image via `multipart/form-data`.
  - Returns JSON response with:
    ```json
    {
      "predictedDigit": 7,
      "confidence": 0.997
    }
    ```

### 4.4 Exception Handling
- Implemented global exception handler (`HandleExceptions.java`):
  - Handles errors like:
    - Missing file upload
    - Invalid image format
    - IO errors
    - General runtime errors
  - Returns structured error response:
    ```json
    {
      "status": 400,
      "error": "Invalid Image",
      "message": "Failed to read the uploaded image"
    }
    ```

### 4.5 Sample Test Images
- Added folder `src/main/resources/test-images`:
  - Contains MNIST-style images for manual API testing.

---

✅ At this stage:
- You can run the API with:
  ```bash
  mvn spring-boot:run
