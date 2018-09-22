import argparse
from image_classifier import *
from image_classifier_util_funcs import *
def main():
    in_args = get_input_args()
    data_dir = in_args.data_dir
    device = 'cpu'
    if(in_args.gpu):
        device = 'cuda'
    
    learning_rate = in_args.learning_rate
    epochs = in_args.epochs
    hidden_layers = in_args.hidden_units
    model_name = in_args.arch
    save_dir = in_args.save_dir
    #Defaulting certain hyper parameters
    drop_p = 0.5
    output_size = 102
    print_every = 40
    image_classifier = ImageClassifier(model_name, learning_rate, print_every, epochs, 
                                    output_size, device, hidden_layers, drop_p, save_dir)
    image_classifier.prepare_model()
    image_classifier.display_model()
    
    image_datasets, dataloaders = get_datasets_and_dataloaders(data_dir)
    image_classifier.train_model(dataloaders['train_loader'], dataloaders['valid_loader'])
    image_classifier.test_model(dataloaders['test_loader'])
    image_classifier.save_to_checkpoint(image_datasets)
    
    
    
def get_input_args():
    # Creates parse 
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, action='store', 
                        help='data directory to pull data from')
    
    parser.add_argument('--save_dir', type=str, default='check_points/', 
                        help='path to folder to save check points')
    parser.add_argument('--arch', type=str, default='vgg19', 
                        help='chosen pretrained model')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--hidden_units', type=int, default=[1000,500], nargs='+', help='Expressed as list Eg: --hidden_units 1000 500')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Numbder of epochs to run')
    
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='Use gpu')
    
    

    # returns parsed argument collection
    return parser.parse_args()

if __name__ == "__main__":
    main()
