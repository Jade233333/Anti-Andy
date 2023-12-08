from PIL import Image, ImageDraw

def draw_rectangles_on_blocks(image_path, image_data, output_path):

    # Open the original image
    image = Image.open(image_path)
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    
    # Iterate over each block in the image data
    for index, row in image_data.iterrows():
        if row['block_num'] != '' and row['level'] in [2] and row['height'] > 10:
            # Get the coordinates of the block
            x = row['left']
            y = row['top']
            width = row['width']
            height = row['height']
            
            # Draw a rectangle on the block
            draw.rectangle([(x, y), (x + width, y + height)], outline='red')
    
    image.save(output_path)



