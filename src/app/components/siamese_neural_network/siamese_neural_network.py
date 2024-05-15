import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from imutils import build_montages
from keras.api.models import Model
from keras.api.layers import (
    Input, 
    Conv2D, 
    Dense, 
    Dropout, 
    GlobalAveragePooling2D, 
    MaxPooling2D, 
    Lambda)
from keras.api.optimizers import Adam
import tf_keras.api._v2.keras.backend as K

from app.utils.constants.constants import *

class SiameseNeuralNetwork(object):
    
    @classmethod
    def classificate_defect(cls, trainX, trainY, testX, testY):
        (trainX, testX) = cls._normalize_data(trainX, testX)
        
        print("[INFO] Preparing positive and negative pairs...")
        (pair_train, label_train) = cls._make_pairs(trainX, trainY)
        print("Train images: {}".format(len(pair_train)))
        print("Train labels: {}".format(len(label_train)))
        (pair_test, label_test) = cls._make_pairs(testX, testY)
        print("Test images: {}".format(len(pair_test)))
        print("Test labels: {}".format(len(label_test)))
        
        model = cls._configure_and_construct_siamese_network(True)
        
        model = cls._compile_model(model, True)
        
        history = cls._train_and_save_model(
            model, pair_train, label_train, pair_test, label_test)
        
        # Plot the training history
        print("[INFO] Plotting training history...")
        cls._plot_training(history)
        
    @classmethod
    def _normalize_data(
            cls, 
            trainX: np.ndarray, 
            testX: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return trainX / 255.0, testX / 255.0
    
    @classmethod
    def _make_pairs(
            cls, 
            images: np.ndarray, 
            labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pairImages = []
        pairLabels = []
        
        # Calculate the total number of classes present in the dataset
        # and then build a list of indexes for each class label that
        # provides the indexes for all examples with a given label
        numClasses = len(np.unique(labels))
        idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
        
        for idxA in range(len(images)):
            # Grab the current image and label belonging to the current
            # iteration
            currentImage = images[idxA]
            label = labels[idxA]
            
            for pair_index in idx[label]:
                if idxA != pair_index:
                    # Pick an image that belongs to the *same* class
                    # label
                    posImage = images[pair_index]
                    
                    # Prepare a positive pair and update the images and labels
                    # lists, respectively
                    pairImages.append([currentImage, posImage])
                    pairLabels.append([1])
            
            for neg_index in np.where(labels != label)[0]:
                if idxA != neg_index:
                    # Grab the indices for each of the class labels *not* 
                    # equal to the current label and randomly pick an image 
                    # corresponding to a label *not* equal to the current label
                    negImage = images[neg_index]
                    
                    # Prepare a negative pair of images and update our lists
                    pairImages.append([currentImage, negImage])
                    pairLabels.append([0])
        
        return (np.array(pairImages), np.array(pairLabels))
    
    @classmethod
    def _build_siamese_architecture(
            cls, 
            inputShape: tuple[int], 
            embeddingDim:int =48, 
            show_summary: bool=False) -> Model:
        # Specify the inputs for the feature extractor network
        inputs = Input(inputShape)
        
        # Define the first set of CONV => RELU => POOL => DROPOUT layers
        x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)
        
        # Second set of CONV => RELU => POOL => DROPOUT layers
        x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        
        # Prepare the final outputs
        pooledOutput = GlobalAveragePooling2D()(x)
        outputs = Dense(embeddingDim)(pooledOutput)
        
        # Build the model
        model = Model(inputs, outputs)
        
        if show_summary:
            model.summary()
        
        return model
    
    @classmethod
    def _euclidean_distance(cls, vectors):
        (featsA, featsB) = vectors
        
        # Compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
            keepdims=True)
        
        # Return the euclidean distance between the vectors
        distance = K.sqrt(K.maximum(sumSquared, K.epsilon()))
        print("distance type", type(distance))
        return distance
    
    @classmethod
    def _configure_and_construct_siamese_network(
            cls, 
            show_summary: bool=False) -> Model:
        # Configure the siamese network
        print("[INFO] Building siamese network...")
        imgA = Input(shape=IMAGE_SHAPE)
        imgB = Input(shape=IMAGE_SHAPE)
        featureExtractor = cls._build_siamese_architecture(
            IMAGE_SHAPE, show_summary)
        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)
        print("featsA type", type(featsA))
        
        # Construct the siamese network
        distance = Lambda(
            cls._euclidean_distance, output_shape=(None, 1))([featsA, featsB])
        outputs = Dense(1, activation="sigmoid")(distance)
        model = Model(inputs=[imgA, imgB], outputs=outputs)
        
        return model
    
    @classmethod
    def _compile_model(
            cls, 
            model: Model, 
            show_summary: bool=False) -> Model:
        print("[INFO] Compiling model...")
        model.compile(loss="binary_crossentropy", optimizer="adam",
            metrics=["accuracy"])
        
        if show_summary:
            model.summary()
            
        return model
    
    @classmethod
    def _train_and_save_model(
            cls, 
            model: Model, 
            pair_train: np.ndarray, 
            label_train: np.ndarray, 
            pair_test: np.ndarray, 
            label_test: np.ndarray):
        print("[INFO] Training model...")
        history = model.fit(
            [pair_train[:, 0], pair_train[:, 1]], 
            label_train, 
            validation_data=([pair_test[:, 0], pair_test[:, 1]], label_test), 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS)
        print("history type", type(history))
        
        # Serialize the model to disk
        print("[INFO] Saving siamese model...")
        model.save(MODEL_PATH)
        
        return history
    
    @classmethod
    def _plot_training(cls, history) -> None:
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(PLOT_PATH)
        
    @classmethod
    def _build_montage(
            cls, 
            pair_train: np.ndarray, 
            label_train: np.ndarray) -> None:
        images = []
        
        # Loop over a sample of our training pairs
        for i in np.random.choice(np.arange(0, len(pair_train)), size=(49,)):
            # Grab the current image pair and label
            imageA = pair_train[i][0]
            imageB = pair_train[i][1]
            label = label_train[i]
            
            imageA = imutils.resize(imageA, width=28, height=28)
            imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            imageB = imutils.resize(imageB, width=28, height=28)
            imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            
            # To make it easier to visualize the pairs and their positive or
            # negative annotations, we're going to "pad" the pair with four
            # pixels along the top, bottom, and right borders, respectively
            output = np.zeros((45, 60), dtype="uint8")
            pair = np.hstack([imageA, imageB])
            output[4:41, 0:56] = pair
            
            # Set the text label for the pair along with what color we are
            # going to draw the pair in (green for a "positive" pair and
            # red for a "negative" pair)
            text = "neg" if label[0] == 0 else "pos"
            color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)
            
            # Create a 3-channel RGB image from the grayscale pair, resize
            # it from 60x36 to 96x51 (so we can better see it), and then
            # draw what type of pair it is on the image
            vis = cv2.merge([output] * 3)
            vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
            cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2)
            
            # Add the pair visualization to our list of output images
            images.append(vis)
            
        montage = build_montages(images, (96, 51), (7, 7))[0]
        
        cv2.imshow("Siamese Image Pairs", montage)
        cv2.waitKey(0)