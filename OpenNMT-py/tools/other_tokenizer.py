from nltk.tokenize.toktok import ToktokTokenizer
import argparse
import tqdm
import codecs
tokenizer = ToktokTokenizer()
# set is_tuple=True if token_classes=True or extra_info=True

# note that paragraphs are allowed to contain newlines

parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る
parser.add_argument('-txt', help='この引数の説明（なくてもよい）')
args = parser.parse_args()

txt=codecs.open(args.txt,"r",encoding='utf8').readlines()

word=codecs.open("word","w")
char=codecs.open("char","w")

#import pdb;pdb.set_trace()
for i in tqdm.tqdm(txt):
    key,value=i.strip().split("@")
    
    tokens = tokenizer.tokenize(value)
    word.write(key+"@"+" ".join(tokens)+"\n")
    char.write(key+"@"+" ".join(list("_".join(tokens)))+"\n")