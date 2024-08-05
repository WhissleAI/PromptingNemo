import re
import sys

def load_tag_file(tag_file):
    tag_dict = {}
    with open(tag_file, 'r') as file:
        for line in file:
            tag, tag_num = line.strip().split()
            tag_dict[tag] = tag_num
    return tag_dict

# Helper function to replace tags in a compound word
def replace_tags(compound_word, tag_to_number):
    # Split by '-' or '_'
    components = re.split(r'[-_]', compound_word)
    
    # Replace each component with its corresponding T<num>
    replaced_components = [f"{tag_to_number.get(component, component)}" for component in components]
    # Join them back with the original separator
    if '-' in compound_word:
        return '-'.join(replaced_components)
    else:
        return '_'.join(replaced_components)


def replace_word_with_tags(infile, tags_file, outfile):
    
    tag_dict = load_tag_file(tags_file)
    
    valid_sents = []
    with open(infile, 'r') as file:
        valid_sents = file.readlines()    
    
    with open(outfile, 'w') as taggedfile:
        for sent in valid_sents:
            
            sent = sent.replace("END.", "END .")
            sent = "TASK TAGGER " + sent + " EOS"
            #print(sent)
            words = sent.split()
            tagged_words = []
            for word in words:

                if '-' in word or '_' in word:
                    tagged_words.append(replace_tags(word, tag_dict))
                elif word in tag_dict:
                    tagged_words.append(tag_dict[word])
                else:
                    tagged_words.append(word)
            
            tagged_sent = " ".join(tagged_words)
        
            #tagged_sent = " ".join(replace_tags(word, tag_dict) for word in words)
            taggedfile.write(tagged_sent+"\n")

if __name__ == "__main__":
    infile = sys.argv[1]
    tags_file = sys.argv[2]
    outfile = sys.argv[3]
    replace_word_with_tags(infile, tags_file, outfile)

