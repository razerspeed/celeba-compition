import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils ,models
from PIL import Image
import torch
import torch.nn as nn

#defining model
model=models.resnet50(pretrained= False)
class CustomResnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        self.model.fc = nn.Identity()
        x = self.fc1(self.model(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def weight_init():
        for block in custom_model.modules():
            if isinstance(block, nn.Linear):
                nn.init.kaiming_uniform_(block.weight)
                nn.init.constant_(block.bias, 1)


custom_model = CustomResnetModel()
custom_model.load_state_dict(torch.load("model_state.pth",map_location=torch.device('cpu')),strict=False)
custom_model.eval()

#preprocess image
class CreateDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.scaler = transforms.Resize([224, 224])
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                              std=[0.5, 0.5, 0.5])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['image']
        img_loc = str(image_name)

        img = Image.open(img_loc)
        img = self.normalize(self.to_tensor(self.scaler(img)))

        return img





def predict(image_file_name):

    image_name=image_file_name
    Dataset = CreateDataset(pd.DataFrame([image_name],columns=['image']))
    ImageDataloader_ResNet = DataLoader(Dataset, batch_size = 1, shuffle=False)

    for img in ImageDataloader_ResNet:
        break

    output = custom_model.forward(img)
    output = output.detach().numpy()
    output=output>0
    output.astype(float)


    return {'male':f'{output[0][0]}','young':f'{output[0][1]}'}







