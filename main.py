"""
Name: Images2Dataset
Author: Rodrigo Loza
Description: This library is intended to convert an image raw dataset to a vectorial structured dataset.
The library assumes:
* Mongodb is running
* The folder structure follows:
                                - Db Folder
                                    - Dataset 0
                                    - Dataset 1
                                    - ...
* All the images have the same size 
"""
# General purpose 
import os
import sys
# Matrix manipulation
import numpy as np 
# Data manipulation
import pandas as pd
# Image manipulation
import cv2
from PIL import Image
# Database
from pymongo import *

class MongoDb:
    """ 
    Connects to mongodb. Instantiates a new database. Supports write, read functions. 
    """
    def __init__(self, 
                name = "data", 
                create = False):
        """
        Constructor
        :param name: string that contains the database's name
        :param create: boolean that decides whether to create a new db or not
        """
        # Connect to the db
        self.client = MongoClient()
        #########TO FIX############
        self.db = client.data
        ###########################
        # If create is true, then start a new array
        if create:
            self.db.data.insert({"_id": 0, "images": []})

    def writeData(self, uri):
        """
        Write field
        : return: confirmation response  
        """
        assert type(uri) == str, "uri must be a string"
        self.db.data.update({"_id": 0}, {"images": uri}) 

    def readData(self, uri):
        """
        NOT IMPLEMENTED YET
        Find filed
        : return: uri
        """
        assert type(uri) == str, "uri must be a string"
        return False

    def readAllData(self):
        """
        Reads all the database
        return: A list that contains all the uris
        """
        return [each for each in self.db.data.find()]

    def dropDb(self):
        """
        Careful!
        Eliminate database
        """
        self.db.data.drop()

class Images2Dataset:
    def __init__(self, 
                dbFolder = os.getcwd(), 
                create = False, 
                db = False, 
                imagesSize = "constant"):
        """
        :param dbFolder: string that contains the folder where all the raw data lives
        :param create: bool that decides whether to create a new db or not
        :param db: bool that decides to use db or not
        :param imagesSize: string that decides to use constant or multiple sizes for images
                            Not constant parameter requires padding feature. 
        """ 
        # Set database folder
        assert dbFolder != os.getcwd(), "dbFolder can't be the same directory"
        assert type(dbFolder) == str, "dbFolder must be a string"
        assert isFolder(dbFolder) == True, "folder does not exist"
        self.dbFolder = dbFolder
        # Set subfolders
        self.subfolders = self.getFolders(self.dbFolder)
        # Set images per folder
        self.images = {}
        for subfolder in self.subfolders:
            self.images[subfolder] = getImages(subfolder)
        # Instantiate db client
        self.db = db
        if db:
            self.dbClient = MongoDb(name = "data", create = create)
        else:
            pass
        # If imagesSize is constant, then get a sample
        if imagesSize == "constant":
            self.height, self.width, self.depth = cv2.imread(self.images[self.subfolders[0]][0])

    def getFolders(self, folder):
        """
        :param folder: string that contains the name of the folder that we want to extract subfolders
        : return: list of subfolders in the folder
        """
        return [os.getcwd()+"/"+folder+"/"+each for each in os.listdir(os.getcwd()+"/"+folder)]

    def getImages(self, folder):
        """ 
        :param folder: string that contains the name of the folder that we want to extract images
        : return: list of images in the folder
        """
        return [os.getcwd()+"/"+self.dbFolder+"/"+folder+"/"+each for each in os.listdir(os.getcwd()+"/"+self.dbFolder+"/"+folder+"/")]

    def uris2MongoDB(self):
        """
        Images stored on hard disk, links available in mongoDB database
        : return: boolean that indicates success 
        """ 
        assert self.db == True, "mongodb can't be used unless db parameter is set to true"
        # Get keys 
        keys = [each for each in self.images.keys()]
        # Save vectors inside keys to db
        for key in keys:
            # Get images of class "key"
            assert key != None, "There was a problem with key"
            imgs = self.images.get(key)
            # Write uris
            for img in imgs:
                self.dbClient.writeData(img)

    def uris2Dataframe(self):
        """
        Convert image uris to dataframe
        : return: pandas dataframe that contains the classes as columns
                  and the uris of each class as rows 
        """ 
        # Create dataframe
        df = pd.DataFrame(self.images)
        return df

    def images2Tensor(self):
        """
        Converted the images in each subfolder to a tensor X and a 
        tensor label Y
        : return: a tensor X of features and a tensor Y of labels
        """
        # Assert memory constraints
        assert (self.height < 50) and (self.width < 50), "Image is too big, try a smaller size"
        # Previous calculations
        rows = self.height*self.width*self.depth
        # Get keys
        keys = [each for each in self.images.keys()]
        columns = []
        for key in keys:
            columns.append(len(self.images.get(key)))
        # Number of classes
        numberClasses = len(self.subfolders)
        # Data (rows are the number of features or pixels) 
        # (columns are the number of examples)
        features = np.zeros([rows,])
        # Labels
        classes = np.eye(numberClasses)
        labels = np.zeros[numberClasses,]
        # Read images in each subfolder
        for k in range(keys):
            # Get images of class "key"
            assert keys[k] != None, "There was a problem with key"
            imgs = self.images.get(keys[k])
            # Read image 
            for img in imgs:
                features = np.c_[features, cv2.imread(img).reshape(-1, 1)]
                labels = np.c_[labels, classes[k].reshape(-1, 1)]
        return features, labels

def isFile(file):
    """
    :param file: string that contains the name of the file we want to test
    : return: boolean that asserts if the file exists 
    """
    if os.path.isfile(file):
        return True
    else:
        return False

def isFolder(folder):
    """
    :param folder: string that contains the name of the folder we want to test
    : return: boolean that asserts if the folder exists 
    """
    if os.path.isfolder(folder):
        return True
    else:
        return False