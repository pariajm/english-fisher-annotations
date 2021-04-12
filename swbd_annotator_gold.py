"""
Pre-processing and annotating Fisher transcripts using 
a SOTA joint parser and disfluency detector model. For 
a complete description of the model, please refer to 
the following paper:
https://www.aclweb.org/anthology/2020.acl-main.346.pdf


* DisfluencyTagger --> finds disfluency labels
* Parser --> finds constituency parse trees
* Annotate --> pre-processes transcripts for annotation

(c) Paria Jamshid Lou, 14th July 2020.
"""

import codecs
import fnmatch
import os
import re   
import torch

import subprocess as sp
import parse_nk

intj = ["uh-huh", "um-hum", "huh-huh", "uh", "um", "huh", "hm", "hum", "oh", "ahah", "ah" , "aha", "eh", "er", "uhah", "huhuh", "mhm", "mm", "hmm", "hm", "em", "hmmm", "yeahhuh", "mmmm", "mmm", "mhmh", "mhm's", "mmhm", "hmhm", "ahha", "uhhum", "shh", "uhhh", "ahh", "umm", "err", "ooh", "ahem", "duh", "aw", "ew", "phew", "shoo", "whoa", "uhoh", "ugh", "phooey", "ouch", "oops", "gee", "ha", "aye", "uhhuh", "umhum", "huhhuh", 'uh-uh', 'hehe', 'uh-oh', 'uhuh', 'huh-uh', 'ahuh', 'um', 'uhm' ] 


class DisfluencyTagger:
    """
    This class is called when self.disfluency==True.    

    Returns:
        A transcript with disfluency labels:
            e.g. "she E she _ likes _ movies _"
            where "E" indicate that the previous 
            word is disfluent and "_" shows that 
            the previous word is fluent.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
    @staticmethod
    def fluent(tokens):
        leaves_tags = []  
        for t in tokens:
            if ")" in t:
                tmp_token = t.replace(")","")
                if tmp_token not in ["'s", "'d", "'re", "'ve", "'m", "'ll", "n't"]: 
                    leaves_tags.append("_")
        return leaves_tags

    @staticmethod
    def disfluent(tokens):
        # remove first and last brackets
        tokens, tokens[-1] = tokens[1:], tokens[-1][:-1]
        open_bracket, close_bracket, pointer = 0, 0, 0      
        df_region = False
        tags = []
        wrds = []
        while pointer < len(tokens):
            open_bracket += tokens[pointer].count("(")                
            close_bracket += tokens[pointer].count(")")
            if "(EDITED" in tokens[pointer]:  
                open_bracket, close_bracket = 1, 0             
                df_region = True
                
            elif ")" in tokens[pointer]:
                label = "E" if df_region else "_" 
                tmp_token = tokens[pointer].replace(")", "")
                if tmp_token in ["'s", "'d", "'re", "'ve", "'m", "'ll", "n't"]:
                    wrds[-1] = wrds[-1]+tmp_token
                else:
                    wrds.append(tmp_token) 
                    tags.append(label)                 
            if all(
                (close_bracket,
                open_bracket == close_bracket)
                ):
                open_bracket, close_bracket = 0, 0
                df_region = False            

            pointer += 1
        # print(wrds)
        # print(tags)
        return tags


class Parser(DisfluencyTagger):
    """
    Loads the pre-trained parser model to find silver parse trees     
   
    Returns:
        Parsed and disfluency labelled transcripts
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def torch_load(self):
        if parse_nk.use_cuda:
            return torch.load(
                self.model
                )
        else:
            return torch.load(
                self.model, 
                map_location=lambda storage, 
                location: storage,
                )

    def run_parser(self, input_sentences):
        eval_batch_size = 1
        print("Loading model from {}...".format(self.model))
        assert self.model.endswith(".pt"), "Only pytorch savefiles supported"

        info = self.torch_load()
        assert "hparams" in info["spec"], "Older savefiles not supported"
        parser = parse_nk.NKChartParser.from_spec(
            info["spec"], 
            info["state_dict"],
            )

        print("Parsing sentences...")
        sentences = [sentence.split() for sentence in input_sentences]
        # Tags are not available when parsing from raw text, so use a dummy tag
        if "UNK" in parser.tag_vocab.indices:
            dummy_tag = "UNK"
        else:
            dummy_tag = parser.tag_vocab.value(0)
        
        all_predicted = []
        for start_index in range(0, len(sentences), eval_batch_size):
            subbatch_sentences = sentences[start_index:start_index+eval_batch_size]
            subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
            predicted, _ = parser.parse_batch(subbatch_sentences)
            del _
            all_predicted.extend([p.convert() for p in predicted])
        
        parse_trees, df_labels = [], []
        for tree in all_predicted:          
            linear_tree = tree.linearize()
            parse_trees.append(linear_tree)
            print(linear_tree)
            if self.disfluency:
                tokens = linear_tree.split() 
                #print(tokens)
                print("\n")
                # disfluencies are dominated by EDITED nodes in parse trees
                if "EDITED" not in linear_tree: 
                    df_labels.append(self.fluent(tokens))
                else:
                    df_labels.append(self.disfluent(tokens))
                    
        return df_labels

           
class Annotate(Parser):   
    """
    Writes parsed and disfluency labelled transcripts into 
    *_parse.txt and *_dys.txt files, respectively.

    """ 
    def __init__(self, **kwargs):
        self.input_path = kwargs["input_path"]
        self.output_path = kwargs["output_path"] 
        self.model = kwargs["model"] 
        self.disfluency = kwargs["disfluency"] 

    def setup(self): 
        all_2004 = self.parse_sentences(            
            trans_data=os.path.join(
                "switchboard_word_alignments", 
                "swb_ms98_transcriptions",
                ),
            parsed_data="gold"
        )

        

    def parse_sentences(self, trans_data, parsed_data):
        input_dir = self.input_path
        output_dir = os.path.join(self.output_path, "gold")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)   
        # Loop over transcription files
        for root, dirnames, filenames in os.walk(input_dir):
            for filename in fnmatch.filter(filenames, "*-asr.wrd"):
                print(filename)
                trans_file = os.path.join(root, filename)
                segments = self.read_transcription(trans_file) 
                # Loop over cleaned/pre-proceesed transcripts         
                doc = [segment.lower() for segment in segments if segment]    
                df_labels = self.run_parser(doc)
                # Write constituency parse trees and disfluency labels into files
                # new_filename = os.path.join(
                #     output_dir, 
                #     os.path.basename(trans_file[:-4])+"_parse.txt"
                #     )
                # with open(new_filename, "w") as output_file:
                #     outputdf_labels_file.write("\n".join(parse_trees))
                #root_path = os.path.join(output_dir, root[-7:-5])
                #if not os.path.exists(root_path):
                    #os.mkdir(root_path)
                #folder_path = os.path.join(root_path, root[-4:])
                #if not os.path.exists(folder_path):
                    #os.mkdir(folder_path)
                new_filename = os.path.join(
                        output_dir, 
                        "train_gold.text"
                        )
                #print(df_labels)
                print(len(df_labels))
                
                with open(trans_file, "r") as fp:
                    ln = fp.readlines()
                print(len(ln))
                with open(new_filename, "w") as ff:
                    for i in range(len(df_labels)):
                        tok = ln[i].strip().split()
                        for t in range(len(tok)):
                            if tok[t].lower() in intj:
                                ff.write(tok[t].upper()+" ")
                            elif df_labels[i][t] == "_":
                                ff.write(tok[t].lower()+" ")
                            elif df_labels[i][t] == "E":
                                ff.write(tok[t].upper()+" ")
                        ff.write("\n")
                #Annotate.write_tags(new_filename, trans_file, df_labels)
                # with open(new_filename, "w") as output_file:
                #     output_file.write("\n".join(df_labels))

        return
    @staticmethod
    def name_changer(filename):
        new_names = []
        args = ["awk", r'{name=substr($1,1,6); gsub("^sw","sw0",name); side=substr($1,7,1); stime=$2; etime=$3;  printf("%s-%s_%06.0f-%06.0f", name, side, int(100*stime+0.5), int(100*etime+0.5)); for(i=4;i<=NF;i++) printf(" %s", $i); printf "\n"}', filename]
        p = sp.Popen(args, stdin = sp.PIPE, stdout = sp.PIPE, stderr = sp.PIPE )
        for l in p.stdout:
            new_names.append(l.decode().split()[0])
        return new_names

    @staticmethod
    def write_tags(new_filename, trans_file, df_labels):
        output_file = open(new_filename, "w") 
        with codecs.open(trans_file, "r", "utf-8") as fp:
            idx = 0
            idxxx = 0
            new_names = Annotate.name_changer(trans_file)
            for line in fp:
            #     if line.startswith("#") or len(line) <= 1:
            #         output_file.write(line+"\n")       
                tokens = line.split()
                
                #print('__________',df_labels)
                #print('***********',line)
                # print("\n"*2)
                if len(tokens[3:]) == 1 and tokens[3] in ["[silence]", "[noise]", "[laughter]", "[vocalized-noise]"]:
                    2>1
                    idxxx += 1
                   #output_file.write(tokens[3]+"\n") 
                elif len(tokens[3:]) == 1 and tokens[3] in intj:
                   idx +=1 
                   idxxx += 1
                elif len(tokens[3:]) == 1 and "<" in tokens[3] and ">" in tokens[3]:
                   idx +=1 
                   idxxx += 1
                elif set(tokens[3:]) < set(["[silence]", "[noise]", "[laughter]", "[vocalized-noise]"]):
                    2>1
                    idxxx += 1
                elif set(tokens[3:]) < set(intj):
                    idx += 1
                    idxxx += 1
                elif set(tokens[3:]) < set(["[silence]", "[noise]", "[laughter]", "[vocalized-noise]"]) ^ set(intj):
                    idxxx+=1
                    idx +=1 
                else: 
                    output_file.write(new_names[idxxx]+" ")                     
                    cntr = 0
                    for w in tokens[3:]:
                        w = w.lower()
                        if w[-2:] == "_1":
                            w = w[:-2]
                        if w in intj:
                            output_file.write(w.upper()+" ")
                            cntr += 1
                        elif "<" in w or ">" in w:
                            cntr += 1
                        elif w in ["[silence]", "[noise]", "[laughter]", "[vocalized-noise]"]:
                            #output_file.write(w+" ") 
                            2>1 
                        
                        elif w[-1] == "-" or w[0] == "-":
                            output_file.write(w.replace("-[","").replace("]-","").replace("-","").replace("[","").replace("]","").upper()+" ")  
                            cntr += 1
                        else:                              
                            lbls = df_labels[idx]
                            #print(lbls)
                            #print(w, cntr)
                            if "[laughter-" in w:
                                w = w.replace("[laughter-", "")
                                w = w.replace("]", "")
                                if w in intj:
                                    output_file.write(w.upper()+" ")
                                if "-" in w[0] or "-" in w[-1]:
                                   output_file.write(w.replace("-", "").upper()+" ")

                                elif lbls[cntr] == "_":
                                    output_file.write(w.lower()+" ")
                                elif lbls[cntr] == "E":
                                    output_file.write(w.upper()+" ")
                                cntr += 1
                            elif "[" in w and "/" in w:
                                didi = w.index("/")
                                w = w[:didi]
                                if "-[" in w or "]-" in w:
                                    w = w.replace("]-","").replace("-[","").replace("[","").replace("]", "")
                                    output_file.write(w.upper()+" ")
                                    cntr += 1
                                else:
                                    if w in intj:
                                       output_file.write(w.upper()+" ")
                                       cntr += 1
                                    elif lbls[cntr] == "_":
                                        output_file.write(w.lower()+' ')
                                        cntr+=1
                                    elif lbls[cntr] == "E":
                                        output_file.write(w.upper() + ' ')
                                        cntr+=1
                            elif "[" in w and w[-1] == "-":
                                w = w.replace("]-","").replace("-[","").replace("]", "").repalce("[","").repalce("-","")
                                output_file.write(w.upper()+" ")
                                
                                cntr += 1
                            elif lbls[cntr] == "_":
                                output_file.write(w.lower()+" ")  
                                cntr += 1
                            elif lbls[cntr] == "E":
                                output_file.write(w.upper()+" ")  
                                cntr += 1
                        # else:
                        #     print(lbls[cntr], w)
                    output_file.write("\n")
                    if cntr != 0:
                        idx += 1
                    idxxx += 1




    def read_transcription(self, trans_file):
        with codecs.open(trans_file, "r", "utf-8") as fp:
            for line in fp:
                           
                tokens = line.split() 
                yield self.validate_transcription(
                    " ".join(tokens)
                    )           

    @staticmethod
    def validate_transcription(label):
        if "[" in label:
            words = label.split()
            for j in range(len(words)):
                if "]-" in words[j]:
                    s_idx = words[j].index("[")
                    words[j] = words[j][:s_idx]+"-"
                if "-[" in words[j]:
                    s_idx = words[j].index("]")
                    words[j] = "-"+words[j][s_idx:]
                if "[" in words[j] and "/" in words[j]:
                    idx = words[j].index("/")
                    words[j] = words[j][idx+1:-1]
            label = " ".join(words)
        
        label = label.replace("_1", " ")
        label = label.replace("_", " ")
        label = re.sub("[ ]{2,}", " ", label)
        label = label.replace("[laughter-", "")
        label = label.replace("[laughter]", "")
        label = label.replace("[noise]", "")
        label = label.replace("[silence]", "")
        label = label.replace("[vocalized-noise]", "")
        label = label.replace(".", "")
        label = label.replace("]-", "-")
        label = label.replace("-[", "-")
        label = label.replace("[", "")
        label = label.replace("]", "")
        label = label.replace(",", "")
        label = label.replace(";", "")
        label = label.replace("?", "")
        label = label.replace("!", "")
        label = label.replace(":", "")
        label = label.replace("\"", "")
        label = label.replace("'re", " 're")
        label = label.replace("'ve", " 've")
        label = label.replace("n't", " n't")
        label = label.replace("'ll", " 'll")
        label = label.replace("'d", " 'd")
        label = label.replace("'m", " 'm")
        label = label.replace("'s", " 's")
        label = label.strip()
        label = label.lower()

        return label if label else None   

        #return label if label else None   
