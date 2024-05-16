import os
import cv2
import h5py
import keras
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
from keras.api.datasets import mnist
import tensorflow as tf
import random
from app.common.common_prints import CommonPrints

from app.utils.constants.constants import *

class SiameseNeuralNetwork(object):
    
    @staticmethod
    def construct_and_train_siamese_neural_network(
            trainX, 
            trainY, 
            testX, 
            testY):
        np.random.seed(0)
        tf.random.set_seed(2)
        random.seed(3)
        
        trainX = trainX / 255.0
        testX = testX / 255.0
        
        print("[INFO] preparing positive and negative pairs...")
        (pair_train, label_train) = SiameseNeuralNetwork.make_pairs_for_training(
            trainX, trainY)
        print(len(pair_train))
        (pair_test, label_test) = SiameseNeuralNetwork.make_pairs_for_training(
            testX, testY)
        print(len(pair_test))
        
        print("[INFO] Building siamese network...")
        imgA = Input(shape=IMAGE_SHAPE)
        imgB = Input(shape=IMAGE_SHAPE)
        featureExtractor = SiameseNeuralNetwork.build_siamese_architecture(IMAGE_SHAPE)
        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)
        
        # Construct the siamese network
        distance = Lambda(SiameseNeuralNetwork.euclidean_distance)([featsA, featsB])
        outputs = Dense(1, activation="sigmoid")(distance)
        model = Model(inputs=[imgA, imgB], outputs=outputs)
        
        print("[INFO] Compiling model...")
        model.compile(
            loss="binary_crossentropy", 
            optimizer="adam",
            metrics=["accuracy"])
        
        model.summary()
        
        print("[INFO] Training model...")
        history = model.fit(
            [pair_train[:, 0], pair_train[:, 1]], label_train[:],
            validation_data=(
                [pair_test[:, 0], pair_test[:, 1]], label_test[:]), 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            shuffle=False)
        
        # Serialize the model to disk
        print("[INFO] Saving siamese model...")
        model.save(os.path.dirname(os.getcwd()) 
                   + MODEL_PATH 
                   + MODEL_NAME)
        
        print("[INFO] plotting training history...")
        SiameseNeuralNetwork.plot_training(history)
        
        return model, history, len(pair_train), len(pair_test)
    
    @staticmethod
    def predict(
            model: Model, 
            original_image: np.ndarray, 
            testX: np.ndarray) -> tuple[float, int]:
        max_probability = 0.0
        max_probability_index = 0
        
        original_image = imutils.resize(original_image, width=120)
        
        original_image = np.expand_dims(original_image, axis=0)
        
        original_image = original_image / 255.0
        testX = testX / 255.0
        
        for index in range(len(testX)):
            test_image = np.expand_dims(testX[index], axis=0)
            preds = model.predict([original_image, test_image])
            probability = preds[0][0]
            
            if probability > max_probability:
                max_probability = probability
                max_probability_index = index
                
        return max_probability, max_probability_index
    
    @staticmethod
    def make_pairs_for_training(
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
    
    @staticmethod
    def build_siamese_architecture(
            inputShape, 
            embeddingDim=48):
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
        
        model.summary()
        
        # Return the model to the calling function
        return model
    
    @staticmethod
    @keras.saving.register_keras_serializable()
    def euclidean_distance(vectors):
        # Unpack the vectors into separate lists
        (featsA, featsB) = vectors
        
        # Compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
            keepdims=True)
        
        # Return the euclidean distance between the vectors
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))
        
    @staticmethod
    def build_montage(
            pairTrain: np.ndarray, 
            labelTrain: np.ndarray) -> None:
        images = []
        
        # Loop over a sample of our training pairs
        for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
            # Grab the current image pair and label
            imageA = pairTrain[i][0]
            imageB = pairTrain[i][1]
            label = labelTrain[i]
            
            # To make it easier to visualize the pairs and their positive or
            # negative annotations, we're going to "pad" the pair with four
            # pixels along the top, bottom, and right borders, respectively
            output = np.zeros((36, 60), dtype="uint8")
            pair = np.hstack([imageA, imageB])
            output[4:32, 0:56] = pair
            
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
            
        # Construct the montage for the images
        montage = build_montages(images, (96, 51), (7, 7))[0]
        
        # Show the output montage
        cv2.imshow("Siamese Image Pairs", montage)
        cv2.waitKey(0)
        
    @staticmethod
    def plot_training(history) -> None:
        """Method to save the history of the trained model

        Parameters:
            history (_type_): History of the trained model
        """
        
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(os.path.dirname(os.getcwd()) 
                    + OUTPUT_DIRECTORY_PATH 
                    + PLOT_NAME)