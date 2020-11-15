"""
This module includes the code for pre-processing and annotating Fisher 
transcripts using a SOTA joint parser and disfluency detector model. 

* DisfluencyTagger --> finds disfluency labels
* Parser --> finds constituency parse trees
* Annotator --> pre-processes transcripts for annotation

(c) Paria Jamshid Lou, 14th July 2020.
"""

import codecs
import fnmatch
import os
import re   
import parse_nk
import torch


class DisfluencyTagger:
    """
    This class is called if self.df_tag == True.    

    Returns:
        A transcript with disfluency labels:
            e.g. "i E i _ like _ movies _"
            where "E" indicates that the previous word is disfluent
            and "_" represents a fluent word.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
    def fluent(self, tokens):
        leaves_tags = []
        for token in tokens:
            if ')' in token:
                leaves_tags.append(token.replace(')','')+' _ ')           
        return ' '.join(leaves_tags)

    def disfluent(self, tokens):
        tokens, tokens[-1] = tokens[1:], tokens[-1][:-1]
        open_bracket, close_bracket, pointer = 0, 0, 0      
        df_region = False
        leaves, tags = [], []
        while pointer < len(tokens):
            open_bracket += tokens[pointer].count("(")                
            close_bracket += tokens[pointer].count(")")
            if '(EDITED' in tokens[pointer]:  
                open_bracket, close_bracket = 1, 0             
                df_region = True
                
            elif ')' in tokens[pointer]:
                leaves.append(tokens[pointer].replace(')', ''))
                label = 'E' if df_region else '_'                   
                tags.append(label)

            if (close_bracket and open_bracket == close_bracket):
                open_bracket,  close_bracket = 0, 0
                df_region = False              

            pointer += 1

        return ' '.join(list(map(lambda t: t[0]+ ' '+t[1], zip(leaves, tags))))


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
            return torch.load(self.model_path)
        else:
            return torch.load(self.model_path, map_location=lambda storage, location: storage)

    def run_parser(self, input_sentences):
        eval_batch_size = len(input_sentences)
        parse_trees, df_labels = [], []
        print("Loading model from {}...".format(self.model_path))
        assert self.model_path.endswith(".pt"), "Only pytorch savefiles supported"

        info = self.torch_load()
        assert 'hparams' in info['spec'], "Older savefiles not supported"
        parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])

        print("Parsing sentences...")
        sentences = [sentence.split() for sentence in input_sentences]

        # Tags are not available when parsing from raw text, so use a dummy tag
        if 'UNK' in parser.tag_vocab.indices:
            dummy_tag = 'UNK'
        else:
            dummy_tag = parser.tag_vocab.value(0)
        
        all_predicted = []
        for start_index in range(0, len(sentences), eval_batch_size):
            subbatch_sentences = sentences[start_index:start_index+eval_batch_size]
            subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
            predicted, _ = parser.parse_batch(subbatch_sentences)
            del _
            all_predicted.extend([p.convert() for p in predicted])
            for tree in all_predicted:   
                linear_tree = tree.linearize()
                parse_trees.append(linear_tree)
                if self.df_tag:
                    tokens = linear_tree.split()
                    if "EDITED" not in linear_tree:
                        df_labels.append(self.fluent(tokens))
                    else:
                        df_labels.append(self.disfluent(tokens))
                    
        return parse_trees, df_labels

           
class Annotator(Parser):   
    """
    Writes parsed and disfluency labelled transcripts into 
    *_parse.txt and *_dys.txt files, respectively.

    """ 
    def __init__(self, **kwargs):
        self.input_path = kwargs['Input_path']
        self.output_path = kwargs['Output_path'] 
        self.model_path = kwargs['Model_path'] 
        self.df_tag = kwargs['Df_tag'] 

    def setup(self): 
        all_2004 = self.parse_sentences(            
            trans_data=os.path.join("LDC2004T19", "fe_03_p1_tran", "data", "trans"),
            parsed_data="fisher-2004-annotations"
        )

        all_2005 = self.parse_sentences(
            trans_data=os.path.join("LDC2005T19", "fe_03_p2_tran", "data", "trans"),
            parsed_data="fisher-2005-annotations"
        )

    def parse_sentences(self, trans_data, parsed_data):
        trans_dir = os.path.join(self.input_path, trans_data)
        target_dir = os.path.join(self.output_path, parsed_data)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)   
        # Loop over transcription files
        for root, dirnames, filenames in os.walk(trans_dir):
            for filename in fnmatch.filter(filenames, "*.txt"):
                doc = []
                trans_file = os.path.join(root, filename)
                segments = self.read_transcription(trans_file) 
                # Loop over cleaned/pre-proceesed transcripts                    
                for segment in segments:            
                    if segment:
                        doc.append(segment)

                # Write constituency parse trees and disfluency labels into files
                new_filename = os.path.join(target_dir, os.path.basename(trans_file[:-4])+'_parse.txt')
                parse_trees, df_labels = self.run_parser(doc)
                with open(new_filename, 'w') as output_file:
                    output_file.write("\n".join(parse_trees))

                if self.df_tag:
                    new_filename = os.path.join(target_dir, os.path.basename(trans_file[:-4])+'_dys.txt')
                    with open(new_filename, 'w') as output_file:
                        output_file.write("\n".join(df_labels))

        return

    def read_transcription(self, trans_file):
        with codecs.open(trans_file, "r", "utf-8") as fp:
            for line in fp:
                if line.startswith("#") or len(line) <= 1:
                    continue                
                tokens = line.split() 
                cleaned_transcript = self.validate_transcription(
                    " ".join(tokens[3:])
                )
                yield cleaned_transcript             

    def validate_transcription(self, label):
        if re.search(r"[0-9]|[(<\[\]&*{]", label) is not None:
            return None

        label = label.replace("_", " ")
        label = re.sub("[ ]{2,}", " ", label)
        label = label.replace(".", "")
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
