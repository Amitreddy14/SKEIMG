import cv2
from scipy.spatial.distance import cosine
from keras.models import Model
from tensorflow.keras.applications import VGG16

class evaluation():
	def __init__(self):
		vgg16 = VGG16(weights='imagenet')
		self.model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
		
	def get_feature_vector(self, img):
		img = cv2.resize(img, (224, 224))
		feature_vector = self.model(img.reshape(1, 224, 224, 3))
		return feature_vector
	
	def calculate_similarity(self, img1, img2):
		vector1 = self.get_feature_vector(img1)
		vector2 = self.get_feature_vector(img2)
		return 1 - cosine(vector1, vector2)
	
	def similarity(self, pred, labels):
		sim = 0
		for i in range(pred.shape[0]):
			sim += self.calculate_similarity(pred[i], labels[i])
		return sim / pred.shape[0]
	def test(self, model, test_input, test_labels):
		sim = 0
		i = 0
		while i + model.batch_size <= test_input.shape[0]:
			batch_input = test_input[i: i+model.batch_size]
			batch_label = test_labels[i: i+model.batch_size]
			pred = model.call(batch_input)
			sim += self.similarity(pred, batch_label)
			i += model.batch_size
		return sim * model.batch_size / i