import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
from torch import __version__
from image_classifier_util_funcs import *

class ImageClassifier:
    
    def __init__(self, model_name='densenet121', learning_rate=None, 
                 print_every=None, epochs=None, output_size=None, 
                 device=None, hidden_layers=None, drop_p=0.5, save_dir='check_points'
                 , load_from_checkpoint=False, checkpoint_loc=None):
        if load_from_checkpoint:
            self.device = device
            self.checkpoint_loc = checkpoint_loc
            self.load_saved_model()
            
        else:
            self.output_size = output_size
            self.hidden_layers = hidden_layers
            self.drop_p = drop_p
            self.epochs = epochs
            self.device = device
            self.model_name = model_name
            self.print_every = print_every
            self.learning_rate = learning_rate
            self.save_dir = save_dir
            
            
            
        
        
        
        
    #Gets pretrained model and adds new classifier to train
    def prepare_model(self):
        print('<-----Preparing the Model Start----->')
        self.model = self.get_pre_trained_model()
        model_name = self.model_name
        output_size = self.output_size
        model = self.model
        drop_p = self.drop_p
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        seq_container = self.model.classifier
        
        input_size = 0
        if model_name.startswith('vgg'):
            input_size = next(iter(seq_container.children())).in_features
        else:
            input_size = model.classifier.in_features
        
        hidden_layers = self.hidden_layers   
        hidden_modules = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden_modules.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        #hidden_modules.extend([nn.Linear(hidden_layers[-1], output_size)])
        ordered_dict=OrderedDict()
        
        #put it in  a sequential container or dictionary
        counter = 1
        for hidden_module in hidden_modules:
            ordered_dict['fc'+str(counter)] = hidden_module
            ordered_dict['relu'+str(counter)] = nn.ReLU()
            ordered_dict['dropout'+str(counter)] = nn.Dropout(drop_p)
            counter += 1
        
        ordered_dict['fc'+str(counter+1)] = nn.Linear(hidden_layers[-1], output_size)
        ordered_dict['output'] = nn.LogSoftmax(dim=1)
        
        #Replace existing classifer in pretrained network with custom classifier
        model.classifier = nn.Sequential(ordered_dict)
        
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)
        print('<-----Preparing the Model End----->')
        
        
        
        
        
    
    def get_pre_trained_model(self):
        model=''
        model_str = self.model_name
        if 'vgg16' == model_str:
            model = models.vgg16(pretrained=True)
        elif 'vgg19' == model_str:
            model = models.vgg19(pretrained=True)
        elif 'densenet121' == model_str:
            model = models.densenet121(pretrained=True)
        else:
            model = models.vgg19(pretrained=True)
    
        return model
        
    def train_model(self, trainloader, validloader):
        device = self.device
        model = self.model
        epochs = self.epochs
        optimizer = self.optimizer
        criterion = self.criterion
        print_every = self.print_every
        steps = 0
        model.to(device)
        print('<-----Training the model start----->')

        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Make sure network is in eval mode for inference
                    model.eval()
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = self.validate_model(validloader)
                
                        print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(test_loss),
                          "Validation Accuracy: {:.3f} %".format(accuracy))
                        #print("Epoch: {}/{}... ".format(e+1, epochs),
                        #  "Loss: {:.4f}".format(running_loss/print_every))
                

                    running_loss = 0
                    model.train()
        
        
        print('<-----Training the model end----->')
    
    def validate_model(self, validloader):
        print('<-----Validating the model start----->')
        
        criterion = self.criterion
        device = self.device
        model = self.model
        test_loss = 0
        correct = 0
        total = 0
        model.eval()
    
        model.to(device)
        with torch.no_grad():
            for data in validloader:
                images, labels = data
                images, labels= images.to(device), labels.to(device)
            
        

                outputs = model.forward(images)
                test_loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        
            accuracy = (100 * correct / total)
        print('<-----Validating the model end----->')    
        return test_loss, accuracy
    
    def test_model(self, testloader):
        print('<-----Testing the model start----->')
        model = self.model
        device = self.device
        correct = 0
        total = 0
        total_num_of_images = 0
        model.eval()
        model.to(device)
        with torch.no_grad():
        
            for data in testloader:
                total_num_of_images+= testloader.batch_size
                images, labels = data
                images, labels= images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the '+str(total_num_of_images)+' test images: %d %%' % (100 * correct / total))
        print('<-----Testing the model end----->')
    
    def display_model(self):
        print(self.model)
    
    def save_to_checkpoint(self, image_datasets):
        print('<-----Saving model to checkpoint start----->')
        self.model.class_to_idx = image_datasets['train_data'].class_to_idx
        checkpoint = { 'epochs': self.epochs,
              
                      'output_size': self.output_size,
                      'model_name': self.model_name,
                      'drop_p': self.drop_p,
                      'hidden_layers': self.hidden_layers,
                      'device': self.device,
                      'lr': self.learning_rate,
                      'print_every': self.print_every,
                      'class_to_idx': self.model.class_to_idx,
                      'state_dict': self.model.state_dict(),
                     }
        save_dir_path = '' 
        if(self.save_dir is not None):
            save_dir_path = self.save_dir
        save_dir_path += self.model_name+'image_classifier.pth'
        torch.save(checkpoint, save_dir_path)
        print('Checkpoint = '+save_dir_path)
        print('<-----Saving model to checkpoint end----->')
        
    def load_saved_model(self):
        print('<-----Loading model from checkpoint start----->')
        #save_dir_path = '' 
        #if(save_dir is not None):
        #    save_dir_path = self.save_dir
        #save_dir_path += self.model_name+'image_classifier.pth'
        save_dir_path = self.checkpoint_loc
        print('Loaded checkpoint = '+save_dir_path)
        check_point = torch.load(save_dir_path)
        self.output_size = check_point['output_size']
        self.hidden_layers = check_point['hidden_layers']
        self.drop_p = check_point['drop_p']
        self.epochs = check_point['epochs']
        self.model_name = check_point['model_name']
        print('Model Name = '+self.model_name)
        self.print_every = check_point['print_every']
        self.learning_rate = check_point['lr']
        
        self.prepare_model()
        self.model.load_state_dict(check_point['state_dict'])
        self.model.class_to_idx = check_point['class_to_idx']
        print('<-----Loading model from checkpoint end----->')
   
    #Returns name of the image and probability prediction
    def predict(self, image_loc, topk=5):
        
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        #np_processed_img = process_image(image_loc)
        img_pil = Image.open(image_loc)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        img_tensor = preprocess(img_pil)
        #device = 'cuda'
        model = self.model
        device = self.device
        if('cuda' == device):
            model = model.to(device)
        model.eval()
        with torch.no_grad():
            #img_tensor = torch.from_numpy(np_processed_img).float()
            #img_tensor = img_tensor.type(torch.DoubleTensor)
            print(type(img_tensor))
            img_tensor = img_tensor.unsqueeze(0)
            if('cuda' == device):
                img_tensor = img_tensor.cuda()
            output = model.forward(img_tensor)
            ps = torch.exp(output)
            #print(ps)
            #print(ps.topk(topk))
            probs , indices = ps.topk(topk)
            #can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            if('cuda' == device):
                probs = probs.cpu()
                indices = indices.cpu()
                
            probs = probs.numpy()[0]
            #print(probs)
            #print(list(map(str,indices.numpy()[0])))
            dict_idx_to_class = {v: k for k, v in model.class_to_idx.items()}
            #print(dict_idx_to_class)
        
            classes = [dict_idx_to_class[x] for x in list(indices.numpy()[0])]
            print(probs)
            print(classes)
        
        return probs, classes
    
    
        
        
  
    