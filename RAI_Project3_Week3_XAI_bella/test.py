from PIL import Image
from load_model import load_model, preprocess

# model = load_model('Cardiomegaly')  # also 'Edema' or 'Atelectasis'
# image = Image.open('images/02_positive_negative/02_01_original.jpg').convert('RGB')
# x = preprocess(image)
# logit = model(x)

# # print(model)
# print(image.size)
# print(x.shape)
# print(logit)

MODELS = ['Cardiomegaly', 'Edema', 'Atelectasis']

for m in MODELS:
    model = load_model(m)
    print(model)