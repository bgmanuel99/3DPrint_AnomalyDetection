import os
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
import tf_keras.api._v2.keras.backend as K

class SiameseNeuralNetwork(object):
    
    @classmethod
    def classificate_defect(cls, trainX, trainY, testX, testY):
        trainX = trainX / 255.0
        testX = testX / 255.0
        
        print("[INFO] Preparing positive and negative pairs...")
        (pairTrain, labelTrain) = cls._make_pairs(trainX, trainY)
        print("Train images: {}".format(len(pairTrain)))
        (pairTest, labelTest) = cls._make_pairs(testX, testY)
        print("Test images: {}".format(len(pairTest)))
        
        IMAGE_SHAPE = (159, 120, 3)
        BATCH_SIZE = 64
        EPOCHS = 5
        
        # configure the siamese network
        print("[INFO] Building siamese network...")
        imgA = Input(shape=IMAGE_SHAPE)
        imgB = Input(shape=IMAGE_SHAPE)
        featureExtractor = cls._build_siamese_architecture(IMAGE_SHAPE)
        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)
        
        # finally, construct the siamese network
        distance = Lambda(
            cls._euclidean_distance, output_shape=(None, 1))([featsA, featsB])
        outputs = Dense(1, activation="sigmoid")(distance)
        model = Model(inputs=[imgA, imgB], outputs=outputs)
        
        # compile the model
        print("[INFO] Compiling model...")
        model.compile(loss="binary_crossentropy", optimizer="adam",
            metrics=["accuracy"])
        model.summary()
        # train the model
        print("[INFO] Training model...")
        history = model.fit(
            [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS)
        
        MODEL_PATH = "{}/data/classification/output/siamese_model.h5".format(
            os.path.dirname(os.getcwd()))
        PLOT_PATH = "{}/data/classification/output/siamese_model_plot.png" \
            .format(os.path.dirname(os.getcwd()))
        
        # Serialize the model to disk
        print("[INFO] saving siamese model...")
        model.save(MODEL_PATH)
        
        # Plot the training history
        print("[INFO] plotting training history...")
        cls._plot_training(history, PLOT_PATH)
        
    @classmethod
    def _normalize_data(cls, trainX, testX):
        pass
    
    @classmethod
    def _make_pairs(cls, images, labels):
        # Initialize two empty lists to hold the (image, image) pairs and
        # labels to indicate if a pair is positive or negative
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
    def _build_siamese_architecture(cls, inputShape, embeddingDim=48):
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
        
        return model
    
    @classmethod
    def _euclidean_distance(cls, vectors):
        (featsA, featsB) = vectors
        
        # Compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
            keepdims=True)
        
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))
    
    @classmethod
    def _plot_training(cls, H, plotPath):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["accuracy"], label="train_acc")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(plotPath)
        
    @classmethod
    def _build_montage(cls, pairTrain, labelTrain):
        images = []
        
        # Loop over a sample of our training pairs
        for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
            # Grab the current image pair and label
            imageA = pairTrain[i][0]
            imageB = pairTrain[i][1]
            label = labelTrain[i]
            
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