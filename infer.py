import torch
from PIL import Image
from torchvision import transforms
import torch
from PIL import Image
from torchvision import transforms
import onnxruntime as ort

from resnet_model import resnet50
from utils.try_gpu import try_gpu

device=try_gpu()

def infer(image,model_file,label_texts):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(image)
    image = torch.unsqueeze(image, 0)  # 图像数据展平
    image = image.to(device)
    model = torch.load(model_file)
    model = model.to(device)
    model.eval()
    output = model(image)
    prob = torch.nn.functional.softmax(output, 1)[0] * 100
    _, indices = torch.sort(output, descending=True)
    with open(label_texts, "rb") as f:
        classes = [line.strip() for line in f.readlines()]
    labels = [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]
    return labels

def onnx_infer(image,model_file,label_texts):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(image)
    image = torch.unsqueeze(image, 0).numpy()  # 图像数据展平
    model = ort.InferenceSession(model_file)
    outputs = model.run(None,{model.get_inputs()[0].name: image})
    outputs = torch.tensor(outputs[0])
    prob = torch.nn.functional.softmax(outputs, 1)[0] * 100
    _, indices = torch.sort(outputs, descending=True)
    with open(label_texts, "rb") as f:
        classes = [line.strip() for line in f.readlines()]
    labels = [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]
    return labels

# def quan_int8_infer(image,model,label_texts):
#     transform = transforms.Compose([
#         transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     image = transform(image)
#     image = torch.unsqueeze(image, 0)  # 图像数据展平
#     image = image.to(device)
#     print("11111")
#     model = model.to(device)
#     model.eval()
#     output = model(image)
#     prob = torch.nn.functional.softmax(output, 1)[0] * 100
#     _, indices = torch.sort(output, descending=True)
#     with open(label_texts, "rb") as f:
#         classes = [line.strip() for line in f.readlines()]
#     labels = [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]
#     return labels


image="data/validation/n3/maque360.png"
image = Image.open(image).convert("RGB")
label_texts="data/bird_label.txt"
# model_file="model/best.pth"
# labels=infer(image,model_file,label_texts)

# model_file="model/best.onnx"
# labels=onnx_infer(image,model_file,label_texts)
#
# model_file="model/quan_int8.pth"
# labels=quan_int8_infer(image,model_file,label_texts)

# for label, prob in labels:
#     print("预测类别为", label, "概率为", prob)