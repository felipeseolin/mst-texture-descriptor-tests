import os

location = '/Users/seolin/Documents/TCC/image-texture-classification-tests/datasets/vistex-texture-dataset/images'
images = sorted(os.listdir(location))
classes = []

for image in images:
    if os.path.isdir(image):
        continue

    class_name = image.split('_')[0]
    if class_name in classes:
        continue

    classes.append(class_name)

print("', '".join(classes))
