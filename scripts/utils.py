import os

def load_env(file_path):
    """
    Load environment variables from a .env file.

    Parameters:
    file_path (str): The path to the .env file.

    Returns:
    None
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and ignore comments and empty lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line into key and value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                # Set the environment variable
                os.environ[key] = value

    print(f"Environment variables loaded from {file_path}")

import yaml

def get_yaml_value(filepath, key):
    """Reads a specified key's value from a YAML file.

    Args:
        filepath: The path to the YAML file.
        key: The key to look for in the YAML file.

    Returns:
        The value associated with the key, or None if the key is not found.
    """
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            return data.get(key)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


import time
import functools

def timer(func):
    """Prints the runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1. Record start time
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2. Record end time
        run_time = end_time - start_time    # 3. Calculate runtime
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

from PIL import Image, ImageDraw

def draw_bounding_box(image_path, bounding_box, box_color="red", box_width=3):
    """
    Draws a bounding box on the image and returns the modified image.
    
    Parameters:
        image_path (str): The path to the image file.
        bounding_box (list or tuple): The bounding box [x1, y1, x2, y2], where
                                      (x1, y1) is the top-left corner and
                                      (x2, y2) is the bottom-right corner.
        box_color (str): The color of the bounding box. Default is 'red'.
        box_width (int): The thickness of the bounding box lines. Default is 3.
    
    Returns:
        PIL.Image.Image: The image with the bounding box drawn on it.
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Ensure bounding_box is valid
        if not (isinstance(bounding_box, (list, tuple)) and len(bounding_box) == 4):
            raise ValueError("Bounding box must be a list or tuple of length 4.")
        
        x1, y1, x2, y2 = bounding_box
        def fix(x,m):
            x = max(x,0)
            x = min(x,m)
            return x
        x1 = fix(x1,image.width)
        x2 = fix(x2,image.width)
        y1 = fix(y1,image.height)
        y2 = fix(y2,image.height)
        if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height or x1 >= x2 or y1 >= y2:
            raise ValueError("Bounding box coordinates are out of image bounds or invalid.")
        
        # Create a drawable object
        draw = ImageDraw.Draw(image)
        
        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
        
        return image
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_img_size(img_path):
    """
    This function takes an image path as input and returns the size of the image. 
    The size is returned as a tuple of (width, height).
    """
    # Open the image file
    img = Image.open(img_path)
    # Get the size of the image
    # The size is returned as a tuple of (width, height)
    size = img.size
    return size

def get_all_img_sizes(df, img_dir):
    """
    This function takes a dataframe and an image directory as input.
    It returns a new dataframe with two additional columns: 'img_width' and 'img_height'.
    These columns contain the width and height of the image, respectively.
    """
    # Create a new dataframe with the same columns as the input dataframe
    df_new = df.copy()
    # Create empty lists to store the width and height of each image
    widths = []
    heights = []
    # Loop through each row of the dataframe
    for _, row in df.iterrows():
        # Get the image path
        img_path = os.path.join(img_dir, row['NAME'])
        # Get the size of the image
        size = get_img_size(img_path)
        # Append the width and height to the respective lists
        widths.append(size[0])
        heights.append(size[1])
    # Add the width and height lists as new columns to the dataframe
    df_new['img_width'] = widths
    df_new['img_height'] = heights
    return df_new

def draw_transparent_shape_top_left_pil(
    img,
    shape="circle",
    shape_size=50,
    transparency=50,
    outline_width=2,
    text_transparency = 50
):
    """
    Draws a small, nearly transparent shape (circle, square, or triangle) in the top-left corner of an image
    and returns the result as a PIL Image.

    Args:
        image_path: Path to the input image.
        shape: The shape to draw ("circle", "square", or "triangle").
        shape_size: Size parameter for the shape (radius for circle, side length for square/triangle).
        transparency: Transparency level of the outline (0-255, 0 is fully transparent, 255 is opaque).
        outline_width: Width of the shape's outline.

    Returns:
        A PIL Image object with the transparent shape drawn on it.
    """
    try:
        # Create a drawing object
        draw = ImageDraw.Draw(img)

        # Define the top-left corner coordinates (with padding)
        padding = 10
        center_x = padding + shape_size
        center_y = padding + shape_size

        # Define the outline color with transparency (RGBA)
        outline_color = (255, 255, 255, transparency)

        # Draw the shape based on the input
        if shape.lower() == "circle":
            bbox = (
                center_x - shape_size,
                center_y - shape_size,
                center_x + shape_size,
                center_y + shape_size,
            )
            draw.ellipse(bbox, outline=outline_color, width=outline_width)

        elif shape.lower() == "square":
            bbox = (
                center_x - shape_size,
                center_y - shape_size,
                center_x + shape_size,
                center_y + shape_size,
            )
            draw.rectangle(bbox, outline=outline_color, width=outline_width)

        elif shape.lower() == "triangle":
            triangle_points = [
                (center_x, center_y - shape_size),  # Top vertex
                (center_x - shape_size, center_y + shape_size),  # Bottom-left vertex
                (center_x + shape_size, center_y + shape_size),  # Bottom-right vertex
            ]
            draw.polygon(triangle_points, outline=outline_color, width=outline_width)

        else:
            #print("Error: Invalid shape specified. Choose from 'circle', 'square', or 'triangle'.")
            return img

        # Return the modified PIL Image
        return img

    except FileNotFoundError:
        print(f"Error: Image file not found at the specified path.")
        return img
    except Exception as e:
        print(f"An error occurred: {e}")
        return img

from PIL import Image, ImageDraw, ImageFont

def write_transparent_text_pil(image, text, font_size_ratio=0.1, transparency=50):
    """
    Writes a text string in the top-left corner of a PIL image using a default font.
    Font size is calculated as a ratio of the image width.
    Returns the result as an RGB PIL Image.
    
    Args:
        image (PIL.Image): Input PIL image.
        text (str): The text string to write.
        font_size_ratio (float, optional): Font size as a ratio of image width. Defaults to 0.1.
        transparency (int, optional): Text transparency level (0-100). Defaults to 50.
        
    Returns:
        PIL.Image: The image with text written on it in RGB format.
        
    Raises:
        ValueError: If font_size_ratio is not between 0 and 1.
        ImportError: If required libraries are not installed.
    """
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    # Input validation
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image")
    if not 0 < font_size_ratio <= 1:
        raise ValueError("font_size_ratio must be between 0 and 1")
    
    # Calculate actual font size based on image width
    font_size = int(image.width * font_size_ratio)
    
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create a transparent overlay for the text
    text_overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_overlay)
    
    # Try to load a default font
    try:
        # Try to use Arial if available
        if os.name == 'nt':  # Windows
            font_path = "arial.ttf"
        else:  # Linux/Mac
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        
        font = ImageFont.truetype(font_path, font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        print("Warning: Using default font as system font not found")
    
    # Add small padding from the edges
    padding = int(image.width * 0.02)  # 2% of image width as padding
    x = padding
    y = padding
    
    # Calculate alpha value (transparency)
    alpha = int((transparency / 100) * 255)
    
    # Draw the text with specified transparency
    draw.text((x, y), text, font=font, fill=(0, 0, 0, alpha))
    
    # Composite the text overlay onto the base image
    result = Image.alpha_composite(image, text_overlay)
    
    # Convert to RGB with white background
    background = Image.new('RGB', result.size, (255, 255, 255))
    background.paste(result, mask=result.split()[3])  # Use alpha channel as mask
    
    return background

import random

def random_flip(image):
    """
    Randomly flip a PIL image horizontally and/or vertically.
    
    Args:
        image (PIL.Image): Input PIL image
        
    Returns:
        PIL.Image: Flipped image
    """
    # Create a copy of the image to avoid modifying the original
    flipped_image = image.copy()
    
    # Randomly decide whether to flip horizontally (left/right)
    if random.choice([True, False]):
        flipped_image = flipped_image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Randomly decide whether to flip vertically (up/down)
    if random.choice([True, False]):
        flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM)
        
    return flipped_image

from PIL import Image

def flip4(image):
    """
    Generates four flipped versions of a PIL image: original, left-right flipped, 
    top-bottom flipped, and both left-right and top-bottom flipped.

    Args:
        image (PIL.Image): Input PIL image.

    Returns:
        list: A list containing four PIL.Image objects:
              [original, flipped_lr, flipped_tb, flipped_both].
    """

    original_image = image.copy()
    flipped_lr = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_tb = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_both = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

    return [original_image, flipped_lr, flipped_tb, flipped_both]

from PIL import Image
import random
import math

def rotate_and_crop_no_padding(image, bbox, angle=None):
    """
    Rotates a PIL Image, crops a square, ensures the original 
    bbox is within the cropped region, and places the crop in the center of the original image.

    Args:
        image: The PIL Image.
        bbox: A tuple (x1, y1, x2, y2) representing the bounding box.
        angle: Optional. Rotation angle in degrees. If None, a random angle between -25 and 25 is used.

    Returns:
        The rotated and cropped PIL Image placed in the center of a new image with original dimensions.
    """
    try:

        x1, y1, x2, y2 = bbox
        original_width, original_height = image.size

        # 1. Rotate the image
        if angle is None:
            angle = random.uniform(-180, 180)
            
        # Rotate, expanding the canvas. No fillcolor specified here
        rotated_image = image.rotate(angle, expand=True) 
        rotated_width, rotated_height = rotated_image.size

        # 2. Rotate the bounding box coordinates
        center_x = original_width / 2
        center_y = original_height / 2
        
        def rotate_point(x, y, angle, cx, cy):
            """Rotates a point around a center."""
            rad = math.radians(angle)
            x -= cx
            y -= cy
            rotated_x = x * math.cos(rad) - y * math.sin(rad)
            rotated_y = x * math.sin(rad) + y * math.cos(rad)
            return rotated_x + cx, rotated_y + cy
        
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        rotated_corners = [rotate_point(x, y, -angle, center_x, center_y) for x, y in corners]

        
        shift_x = (rotated_width - original_width)/2
        shift_y = (rotated_height - original_height)/2
        rotated_corners = [(x+shift_x, y+shift_y) for x,y in rotated_corners]
        
        min_x = min(x for x, y in rotated_corners)
        min_y = min(y for x, y in rotated_corners)
        max_x = max(x for x, y in rotated_corners)
        max_y = max(y for x, y in rotated_corners)

        # 3. Determine the square crop region
        crop_size = max(max_x - min_x, max_y - min_y)

        crop_x1 = max(0, min_x - (crop_size - (max_x-min_x))/2) 
        crop_y1 = max(0, min_y - (crop_size - (max_y-min_y))/2)
        crop_x2 = min(rotated_width, crop_x1 + crop_size)
        crop_y2 = min(rotated_height, crop_y1 + crop_size)

        if crop_size >= rotated_width:
            crop_x1 = 0
            crop_x2 = rotated_width
        if crop_size >= rotated_height:
            crop_y1 = 0
            crop_y2 = rotated_height

        cropped_image = rotated_image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))

        # 4. Place the crop in the center of the original image
        # Use a copy of the original image to avoid modifying the original directly
        result_image = image.copy()
        
        # Calculate paste coordinates to center the cropped image
        paste_x = (original_width - cropped_image.width) // 2
        paste_y = (original_height - cropped_image.height) // 2

        result_image.paste(cropped_image, (paste_x, paste_y))
    except:
        return image
    return result_image

def weighted_random_selection(data):
    """
    Randomly selects a key from a dictionary based on weighted probabilities.

    Args:
        data: A dictionary where keys are strings and values are positive integers 
              representing weights.

    Returns:
        A randomly selected key (string) from the dictionary.
    """
    if len(data) == 0:
        return None
    if len(data) == 1:
        return [key for key in data.keys()][0]

    total_weight = sum(data.values())

    if total_weight <= 0:
        raise ValueError("Total weight must be greater than zero")
        
    random_num = random.uniform(0, total_weight)
    cumulative_weight = 0

    for key, weight in data.items():
        cumulative_weight += weight
        if random_num <= cumulative_weight:
            return key

    # Should not reach here unless there's a rounding error
    return [key for key in data.keys()][0]

def combine_cropped_images(img1, bbox1, img2, bbox2):
    """
    Crops two images based on bounding boxes, resizes the first cropped image
    to match the second, and then combines them, placing the first over the second.

    Args:
        img1: PIL Image object representing the first image.
        bbox1: Tuple (x1, y1, x2, y2) representing the bounding box for img1.
        img2: PIL Image object representing the second image.
        bbox2: Tuple (x1, y1, x2, y2) representing the bounding box for img2.

    Returns:
        A PIL Image object representing the combined image.
    """

    # Crop the images based on bounding boxes
    cropped_img1 = img1.crop(bbox1)
    cropped_img2 = img2.crop(bbox2)

    # Resize cropped_img1 to match the size of cropped_img2
    cropped_img1 = cropped_img1.resize(cropped_img2.size)

    # Create a new image with the same size as img2
    combined_image = img2.copy()

    # Paste cropped_img1 onto the new image at the location defined by bbox2
    combined_image.paste(cropped_img1, bbox2[:2])

    return combined_image

