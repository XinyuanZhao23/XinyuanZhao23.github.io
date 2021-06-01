#2.1

#1
from torchvision import models

#2
print(dir(models))

#3
alexnet = models.AlexNet()

#4

resnet = models.resnet18(pretrained=True)

#5
resnet

#6
from torchvision import transforms
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)])


#7
from PIL import Image
img = Image.open("123.jpg")

#8

img.show()


#9
img_t = preprocess(img)

#10
import torch
batch_t = torch.unsqueeze(img_t,0)

#11

resnet.eval()


#2.2

#2
netG = ResNetGenerator()

#3
model_path = '../data/p1ch2/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)

#4
netG.eval()

#5
from PIL import Image
from torchvision import transforms

#6
preprocess = transforms.Compose([transforms.Resize(256),
transforms.ToTensor()])

#7
img = Image.open("../data/p1ch2/horse.jpg")
img

#8
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# In[9]:
batch_out = netG(batch_t)

# In[10]:
out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
# out_img.save('../data/p1ch2/zebra.jpg')
out_img
# Out[10]:
<PIL.Image.Image image mode=RGB size=316x256 at 0x23B24634F98>

