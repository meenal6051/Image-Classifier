import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import copy

# Define available architectures and their corresponding hidden units
ARCHITECTURES = {
    'vgg16': 25088,
    'densenet121': 1024,
    'alexnet': 9216,
}

# Initialize variables with default values
arch = 'vgg16'
hidden_units = 512
learning_rate = 0.0001
epochs = 1
device = 'cpu'

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Directory with data for image classifier to train and test')
parser.add_argument('-a', '--arch', action='store', type=str, help='Choose from vgg16, alexnet, or densenet121')
parser.add_argument('-H', '--hidden_units', action='store', type=int, help='Number of hidden units for 1st layer')
parser.add_argument('-l', '--learning_rate', action='store', type=float, help='Learning rate for the model')
parser.add_argument('-e', '--epochs', action='store', type=int, help='Number of epochs for gradient descent')
parser.add_argument('-s', '--save_dir', action='store', type=str, help='Name of the file to save the trained model')
parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

# Validate architecture input
if args.arch:
    if args.arch not in ARCHITECTURES:
        print("Invalid architecture choice. Choose from vgg16, alexnet, or densenet121.")
        exit(1)

    arch = args.arch
    hidden_units = ARCHITECTURES[args.arch]

# Check for other command-line arguments and update variables accordingly
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to create the model and classifier
def create_model(arch, hidden_units, learning_rate):
    model = getattr(models, arch)(pretrained=True)
    
    # Freeze feature parameters so as not to backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Build the classifier for the model
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(ARCHITECTURES[arch], hidden_units)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(p=0.15)),
        ('fc2', nn.Linear(hidden_units, 512)),
        ('ReLu2', nn.ReLU()),
        ('Dropout2', nn.Dropout(p=0.15)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1, last_epoch=-1)

    return model, criterion, optimizer, scheduler

# Create the model, criterion, optimizer, and scheduler
model, criterion, optimizer, scheduler = create_model(arch, hidden_units, learning_rate)

print("-" * 10)
print(f"Your model with {arch} architecture has been built!")

# Define data directories for training, validation, and testing
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define data transforms for training and validation sets and normalize images
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets for training and validation
image_datasets = {x: datasets.ImageFolder(data_dir + f'/{x}', transform=data_transforms[x])
                  for x in ['train', 'valid']}

# Create dataloaders for training and validation
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
               for x in ['train', 'valid']}

# Define dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

# Define class names
class_names = image_datasets['train'].classes

# Function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('-' * 10)
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best valid Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# Train the model
model_trained = train_model(model, criterion, optimizer, scheduler, num_epochs=epochs)

print('-' * 10)
print(f'Your {arch} model has been successfully trained')
print('-' * 10)

# Function to save the model
def save_model(model, save_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
    }

    save_path = save_dir if save_dir else 'checkpoint.pth'
    torch.save(checkpoint, save_path)

# Save the model
save_model(model_trained, args.save_dir)
print('-' * 10)
print(f'Your {arch} model has been successfully saved.')
print('-' * 10)

