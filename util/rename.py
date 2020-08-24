import os

location = '/home/seolin/Documents/KylbergTextureDataset-1/wall1-without-rotations'
class_name = 'WAL'
images = os.listdir(location)

print('Começando a renomear....')
for acc, image in enumerate(images):
    new_name = f"{class_name}_{str(acc+1).zfill(6)}.png"
    print(image + ' -> ' + new_name)
    os.rename(f"{location}/{image}", f"{location}/{new_name}")

print("Finalizou!!")

# ❯ mogrify -format jpg *.ppm 
