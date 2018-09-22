import argparse
from image_classifier import *
from image_classifier_util_funcs import *
def main():
    in_args = get_input_args()
    #print(in_args)
    image_loc = in_args.image_loc
    checkpoint_loc = in_args.checkpoint
    top_k = in_args.top_k
    category_names = in_args.category_names
    device = 'cpu'
    if(in_args.gpu):
        device = 'cuda'
    image_classifier = ImageClassifier(device=device, load_from_checkpoint=True, checkpoint_loc=checkpoint_loc)
    #image_classifier.display_model()
    probs, classes = image_classifier.predict(image_loc , top_k)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    class_name = cat_to_name[classes[0]]
    class_names = [cat_to_name[x] for x in classes]
    print('Predicted Flower Name = {}, Probability = {}'.format(class_name,probs[0]))
    print('Top {} predicted classes are {}'.format(top_k,class_names))
def get_input_args():
    # Creates parse 
    parser = argparse.ArgumentParser()

    parser.add_argument('image_loc', type=str, action='store', 
                        help='Location of image')
    parser.add_argument('checkpoint', type=str, action='store', 
                        help='Location of checkpoint')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='Use gpu')
    parser.add_argument('--top_k', type=int, default=3, 
                        help='Top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', 
                        help='Category to class name json ')
    
    
    
    
    

    # returns parsed argument collection
    return parser.parse_args()
if __name__ == "__main__":
    main()