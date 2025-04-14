import cv2
from scipy.spatial.distance import cosine
from keras.models import Model
from tensorflow.keras.applications import VGG16

class evaluation():
	def __init__(self):
		vgg16 = VGG16(weights='imagenet')
		self.model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)