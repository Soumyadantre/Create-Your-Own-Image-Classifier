import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import numpy as np
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import json

def data_transformation(args):
    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")

    # paths Validation 
    if not os.path.exists(args.data_directory):
        print("Data Directory doesn't exist: {}".format(args.data_directory))
        raise FileNotFoundError
    if not os.path.exists(args.save_directory):
        print("Save Directory doesn't exist: {}".format(args.save_directory))
        raise FileNotFoundError

    if not os.path.exists(train_dir):
        print("Train folder doesn't exist: {}".format(train_dir))
        raise FileNotFoundError
    if not os.path.exists(valid_dir):
        print("Valid folder doesn't exist: {}".format(valid_dir))
        raise FileNotFoundError

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])

    train_data = ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=valid_transforms)

    train_data_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data_loader = data.DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_data_loader, valid_data_loader, train_data.class_to_idx

def train_model(args, train_data_loader, valid_data_loader, class_to_idx):
       
    if args.model_arch == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif args.model_arch == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif args.model_arch == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif args.model_arch == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)

    
    for param in model.parameters():
        param.requires_grad = False

    in_features_of_pretrained_model = model.classifier[0].in_features

    # Alter the classifier to have 102 output features (corresponding to the number of classes)
    classifier = nn.Sequential(nn.Linear(in_features=in_features_of_pretrained_model, out_features=2048, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Dropout(p=0.2),
                               nn.Linear(in_features=2048, out_features=102, bias=True),
                               nn.LogSoftmax(dim=1)
                              )

    model.classifier = classifier

    # model training
    criterion = nn.NLLLoss()  # Specify the criterion
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)  # Specify the optimizer for the classifier parameters

   
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead.")
    else:
        device = 'cpu'
    print("Using {} to train the model.".format(device))
            
    
    model.to(device)

   
    print_every = 20

    # For each epoch
    for e in range(args.epochs):
        step = 0
        running_train_loss = 0
        running_valid_loss = 0

        
        for images, labels in train_data_loader:
            step += 1

           
            model.train()

            
            images, labels = images.to(device), labels.to(device)

           
            optimizer.zero_grad()

            
            outputs = model(images)

           
            train_loss = criterion(outputs, labels)

           
            train_loss.backward()

            
            optimizer.step()

            running_train_loss += train_loss.item()

            if step % print_every == 0 or step == 1 or step == len(train_data_loader):
                print("Epoch: {}/{} Batch % Complete: {:.2f}%".format(e+1, args.epochs, (step)*100/len(train_data_loader)))

       

        model.eval()
        with torch.no_grad():
            print("Validating Epoch....")
            
            running_accuracy = 0
            running_valid_loss = 0
            for images, labels in valid_data_loader:

                
                images, labels = images.to(device), labels.to(device)

                
                outputs = model(images)

               
                valid_loss = criterion(outputs, labels)
                running_valid_loss += valid_loss.item()

                # Accuracy 
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # statistics
            average_train_loss = running_train_loss/len(train_data_loader)
            average_valid_loss = running_valid_loss/len(valid_data_loader)
            accuracy = running_accuracy/len(valid_data_loader)
            print("Train Loss: {:.3f}".format(average_train_loss))
            print("Valid Loss: {:.3f}".format(average_valid_loss))
            print("Accuracy: {:.3f}%".format(accuracy*100))

    model.class_to_idx = class_to_idx
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'epochs': args.epochs,
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'vgg_type': args.model_arch
                 }

    torch.save(checkpoint, os.path.join(args.save_directory, "checkpoint.pth"))
    print("Model saved to {}".format(os.path.join(args.save_directory, "checkpoint.pth")))
    return True

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(dest='data_directory', help="Specify the directory of the training images. For example, if a sample file is in /flowers/train/daisy/001.png, then supply /flowers. Expect two folders within, 'train' and 'valid'.")

    
    parser.add_argument('--save_directory', dest='save_directory', help="Specify the directory where the model will be saved after training.", default='../saved_models')
    parser.add_argument('--learning_rate', dest='learning_rate', help="Specify the learning rate when training the model. The default is 0.003. Expect a float type.", default=0.003, type=float)
    parser.add_argument('--epochs', dest='epochs', help="Specify the number of epochs when training the model. The default is 5. Expect an integer type.", default=3, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to train the model on the GPU via CUDA", action='store_true')
    parser.add_argument('--model_arch', dest='model_arch', help="Specify the type of pre-trained model that will be used", default="vgg19", type=str, choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])

    
    args = parser.parse_args()

    
    train_data_loader, valid_data_loader, class_to_idx = data_transformation(args)

    
    train_model(args, train_data_loader, valid_data_loader, class_to_idx)
