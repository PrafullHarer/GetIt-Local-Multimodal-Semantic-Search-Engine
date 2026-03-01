
from PIL import Image
from models import generate_caption, get_image_embedding

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    caption = generate_caption(image)
    embedding = get_image_embedding(image)
    return caption, embedding
