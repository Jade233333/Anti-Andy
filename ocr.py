import keras_ocr
import matplotlib.pyplot as plt

# Get the pipeline from keras-ocr, which is a convenience wrapper that combines the detector and recognizer models into a single pipeline.
pipeline = keras_ocr.pipeline.Pipeline()

# Use the pipeline to recognize text from an image. You can use any image file format that is supported by keras (e.g., png, jpg).
image = keras_ocr.tools.read('path_to_your_image.jpg') 
predictions = pipeline.recognize([image])[0]

# Display the image and the predicted text
fig, ax = plt.subplots()
keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
