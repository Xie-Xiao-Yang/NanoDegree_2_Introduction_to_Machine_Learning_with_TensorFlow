import argparse
import my_utils
import json


parser = argparse.ArgumentParser(description='predict a image')
parser.add_argument('image_path', help='path to the image you want to predict')
parser.add_argument('model_name', help='model name')
parser.add_argument('--top_k',type=int, dest='topk', help='top-k value')
parser.add_argument('--category_names', dest='category_names', help='category file name')

args = parser.parse_args()


if args.topk == None:
    topk = 5
else:
    topk = args.topk
    
if args.category_names == None:
    category_names = 'label_map.json'
else:
    category_names = args.category_names

class_names = {}
with open(category_names, 'r') as f:
    class_names = json.load(f)

probs, classes = my_utils.predict(args.image_path, args.model_name, topk)
for i in range(topk):
    print("{} : {}".format(class_names[str(classes[i]+1)], probs[i]))
