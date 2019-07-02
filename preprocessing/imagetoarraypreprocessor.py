from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat # channels_first or channels_last (string data type)

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of the image

        # it returns arranged as a NumPy array
        return img_to_array(image, data_format=self.dataFormat)