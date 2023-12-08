import keras_ocr

# Get the pipeline from keras-ocr, which is a convenience wrapper that combines the detector and recognizer models into a single pipeline.
pipeline = keras_ocr.pipeline.Pipeline()

# Use the pipeline to recognize text from an image. You can use any image file format that is supported by keras (e.g., png, jpg).
image = keras_ocr.tools.read('processed_image.png') 
predictions = pipeline.recognize([image])[0]

for text, box in predictions:
    print(text)

recognized_text = ' '.join([text for text, box in predictions])
print(recognized_text)
