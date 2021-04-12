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

import parse_nk


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
        leaves_tags = [t.replace(")","")+" _" for t in tokens if ")" in t]      
        return " ".join(leaves_tags)

    @staticmethod
    def disfluent(tokens):
        # remove first and last brackets
        tokens, tokens[-1] = tokens[1:], tokens[-1][:-1]
        open_bracket, close_bracket, pointer = 0, 0, 0      
        df_region = False
        tags = []
        while pointer < len(tokens):
            open_bracket += tokens[pointer].count("(")                
            close_bracket += tokens[pointer].count(")")
            if "(EDITED" in tokens[pointer]:  
                open_bracket, close_bracket = 1, 0             
                df_region = True
                
            elif ")" in tokens[pointer]:
                label = "E" if df_region else "_"  
                tags.append(
                    (tokens[pointer].replace(")", ""), label)
                    )                 
            if all(
                (close_bracket,
                open_bracket == close_bracket)
                ):
                open_bracket, close_bracket = 0, 0
                df_region = False            

            pointer += 1
        return " ".join(list(map(lambda t: " ".join(t), tags)))


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
            if self.disfluency:
                tokens = linear_tree.split()
                # disfluencies are dominated by EDITED nodes in parse trees
                if "EDITED" not in linear_tree: 
                    df_labels.append(self.fluent(tokens))
                else:
                    df_labels.append(self.disfluent(tokens))
                    
        return parse_trees, df_labels

           
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
            parsed_data="swbd-annotations"
        )

        

    def parse_sentences(self, trans_data, parsed_data):
        input_dir = os.path.join(self.input_path, trans_data)
        output_dir = os.path.join(self.output_path, parsed_data)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)   
        # Loop over transcription files
        for root, dirnames, filenames in os.walk(input_dir):
            for filename in fnmatch.filter(filenames, "*trans.text"):
                trans_file = os.path.join(root, filename)
                segments = self.read_transcription(trans_file) 
                # Loop over cleaned/pre-proceesed transcripts         
                doc = [segment for segment in segments if segment]    
                parse_trees, df_labels = self.run_parser(doc)
                # Write constituency parse trees and disfluency labels into files
                new_filename = os.path.join(
                    output_dir, 
                    os.path.basename(trans_file[:-4])+"_parse.txt"
                    )
                with open(new_filename, "w") as output_file:
                    output_file.write("\n".join(parse_trees))

                if self.disfluency:
                    new_filename = os.path.join(
                        output_dir, 
                        os.path.basename(trans_file[:-4])+"_dys.txt"
                        )
                    with open(new_filename, "w") as output_file:
                        output_file.write("\n".join(df_labels))

        return

    def read_transcription(self, trans_file):
        with codecs.open(trans_file, "r", "utf-8") as fp:
            for line in fp:
                if line.startswith("#") or len(line) <= 1:
                    continue                
                tokens = line.split() 
                yield self.validate_transcription(
                    " ".join(tokens[3:])
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
        label = label.replace("_", " ")
        label = re.sub("[ ]{2,}", " ", label)
        label = label.replace("[laughter-", "")
        label = label.replace("[laughter]", "")
        label = label.replace("[noise]", "")
        label = label.replace("[silence]", "")
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

        return label if label else None   
