import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext('/Users/jade/Developer/Anti-Andy/test/3.jpg')
print(result)