import pickle
from PIL import Image

with open("/mnt/ssd_data/hil-serl/examples/classifier_data/ram_insertion_2_success_images_2025-09-11_09-39-00.pkl", "rb") as f:
    data = pickle.load(f)
    
img = Image.fromarray(data[0]['observations']['images']['wrist_1'])
img.save("./classifier_data/wrist_1.png")

img = Image.fromarray(data[0]['observations']['images']['wrist_2'])
img.save("./classifier_data/wrist_2.png")
