import heapq
from PIL import Image
from torchvision import transforms
from vgg import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

tran = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

im = './1.jpg'
im = Image.open(im)
im = tran(im)
im.unsqueeze_(dim=0)
print("input image info:", im.shape)

out = vgg(im)
outnp = out.data[0].numpy()
res = heapq.nlargest(5, range(len(outnp)), outnp.take)
print("top-5 results:", res)
print("output tensor shape:", out.shape)
