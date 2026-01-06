from PIL  import Image
from matplotlib  import pyplot as plt
image_path = "/m2v_intern/mengzijie/data/example_video_dataset/wans2v/pose.png"
height = 256
width = 192

from PIL import ImageOps
input_image = Image.open(image_path).convert("RGB")
input_image = ImageOps.fit(input_image, (width, height), Image.LANCZOS)



plt.imsave("output.png",  input_image)
