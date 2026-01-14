from PIL import Image
Image.open(r"C:\SLM\python_code\images\einstein_.bmp").convert("RGB") \
    .quantize(colors=256, method=Image.FASTOCTREE, dither=Image.FLOYDSTEINBERG) \
    .save("C:\SLM\python_code\images\einstein_8.bmp", format="BMP", bits=8)
