import cv2
import imutils
import numpy as np
from imutils import build_montages
from keras.api.models import Model
from keras.api.layers import (
    Input, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D)
import tf_keras.api._v2.keras.backend as K
import matplotlib.pyplot as plt

class SiameseNeuralNetwork(object):
    
    @classmethod
    def classificate_defect(cls):
        pass
    
    @classmethod
    def make_pairs(cls, images, labels):
        # initialize two empty lists to hold the (image, image) pairs and
        # labels to indicate if a pair is positive or negative
        pairImages = []
        pairLabels = []
        
        # calculate the total number of classes present in the dataset
        # and then build a list of indexes for each class label that
        # provides the indexes for all examples with a given label
        numClasses = len(np.unique(labels))
        idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
        
        # loop over all images
        for idxA in range(len(images)):
            # grab the current image and label belonging to the current
            # iteration
            currentImage = images[idxA]
            label = labels[idxA]
            
            # randomly pick an image that belongs to the *same* class
            # label
            idxB = np.random.choice(idx[label])
            posImage = images[idxB]
            
            # prepare a positive pair and update the images and labels
            # lists, respectively
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])
            
            # grab the indices for each of the class labels *not* equal to
            # the current label and randomly pick an image corresponding
            # to a label *not* equal to the current label
            negIdx = np.where(labels != label)[0]
            negImage = images[np.random.choice(negIdx)]
            
            # prepare a negative pair of images and update our lists
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])
            
        # return a 2-tuple of our image pairs and labels
        return (np.array(pairImages), np.array(pairLabels))
    
    @classmethod
    def _build_siamese_architecture(cls, inputShape, embeddingDim=48):
        # specify the inputs for the feature extractor network
        inputs = Input(inputShape)
        
        # define the first set of CONV => RELU => POOL => DROPOUT layers
        x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)
        
        # second set of CONV => RELU => POOL => DROPOUT layers
        x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        
        # prepare the final outputs
        pooledOutput = GlobalAveragePooling2D()(x)
        outputs = Dense(embeddingDim)(pooledOutput)
        
        # build the model
        model = Model(inputs, outputs)
        
        # return the model to the calling function
        return model
    
    @classmethod
    def _euclidean_distance(cls, vectors):
        # unpack the vectors into separate lists
        (featsA, featsB) = vectors
        
        # compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
            keepdims=True)
        
        # return the euclidean distance between the vectors
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))
    
    @classmethod
    def _plot_training(cls, H, plotPath):
        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["val_loss"], label="val_loss")
        plt.plot(H.history["accuracy"], label="train_acc")
        plt.plot(H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(plotPath)
        
    @classmethod
    def _build_montage(cls, pairTrain, labelTrain):
        # initialize the list of images that will be used when building our
        # montage
        images = []
        # loop over a sample of our training pairs
        for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
            # grab the current image pair and label
            imageA = pairTrain[i][0]
            imageB = pairTrain[i][1]
            label = labelTrain[i]
            
            imageA = imutils.resize(imageA, width=28, height=28)
            imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            imageB = imutils.resize(imageB, width=28, height=28)
            imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            
            # to make it easier to visualize the pairs and their positive or
            # negative annotations, we're going to "pad" the pair with four
            # pixels along the top, bottom, and right borders, respectively
            output = np.zeros((45, 60), dtype="uint8")
            pair = np.hstack([imageA, imageB])
            output[4:41, 0:56] = pair
            
            # set the text label for the pair along with what color we are
            # going to draw the pair in (green for a "positive" pair and
            # red for a "negative" pair)
            text = "neg" if label[0] == 0 else "pos"
            color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)
            
            # create a 3-channel RGB image from the grayscale pair, resize
            # it from 60x36 to 96x51 (so we can better see it), and then
            # draw what type of pair it is on the image
            vis = cv2.merge([output] * 3)
            vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
            cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2)
            
            # add the pair visualization to our list of output images
            images.append(vis)
            
        # construct the montage for the images
        montage = build_montages(images, (96, 51), (7, 7))[0]
        # show the output montage
        cv2.imshow("Siamese Image Pairs", montage)
        cv2.waitKey(0)