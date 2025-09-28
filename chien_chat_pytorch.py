import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor

"""
    Mon programme construit un Convolutional Neural Network qui utilisent des filtres pour trouver des motifs ce qui permet au modèle d'identifier si la photo représente un chien ou un chat.
    J'ai utilsé la base de données Cats and Dogs Classification Dataset disponible sur Kaggle : https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
    Après avoir récupéré ce dataset, j'ai crée 2 dossiers Train et Test qui contiennent chacun deux dossiers cats et dogs.
"""

transform = transforms.Compose([
    transforms.Resize((128, 128)),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                         std=[0.5, 0.5, 0.5])  
])

train_data = datasets.ImageFolder(root="data/Chien_Chat/Train", transform=transform)
test_data  = datasets.ImageFolder(root="data/Chien_Chat/Test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32)

print("Classes :", train_data.classes)

if torch.cuda.is_available():
    device = torch.device("cuda")
print("Device is ",device)

class ChienOrChat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(3,16,kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                
                nn.Conv2d(16,32,kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                
                nn.Conv2d(32,64,kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
            )
        
        self.fc_layer = nn.Sequential(
                nn.Linear(64*16*16, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        self.flatten= nn.Flatten()
        
    def forward(self,x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        return x
    
model = ChienOrChat().to(device)
loss_fc = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=30
for i in range(0,epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        loss = loss_fc(pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("Epoch ",i+1)
    print("Loss :", loss.item())
        
model.eval()
correct, test_loss =0, 0
size_data = len(test_data)
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        test_loss += loss_fc(pred, y_batch).item()
        correct += (pred.argmax(1) == y_batch).type(torch.float).sum().item()
    
print("Accuracy finale : "+str((correct/size_data)*100) + "%")
print("Avg Loss : ", test_loss/len(test_loader))

    
