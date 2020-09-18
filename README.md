# Fisher Parse Trees and Disfluency Labels
This repo contains the code for annotating English Fisher Speech Transcripts. Since Fisher Corpus is not open-source, we cannot release the annotated transcripts. We instead provide the recipe for pre-processing and annotating Fisher transcripts. The annotations include silver constituency parse trees and silver disfluency labels which are allocated using a state-of-the-art joint parsing and disfluency detection model (with parsing accuracy of 93.9% and disfluency detection f-score of 92.4% on Switchboard dev set), as described in [Improving Disfluency Detection by Self-Training a Self-Attentive Model](https://www.aclweb.org/anthology/2020.acl-main.346/) from ACL 2020. Since a disfluency tag is allocated to each word, you can use this recipe to obtain the fluent English Fisher transcripts (by removing the words tagged as disfluent).

### Using the model to annotate Fisher 
Running the following commands, you will end up with two types of output: 
* ```fe_**_****_parse.txt``` which includes Fisher constituency parse trees
* ```fe_**_****_dys.txt``` which contains Fisher disfluency labelled transcripts (where *_* and *E* indicate that the previous word is fluent or disfluent, respectively). Remove the words tagged as *E* to obtain the fluent version of transcripts.

```
$ git clone https://github.com/pariajm/fisher-annotations
$ cd fisher-annotations
$ mkdir model && cd model
$ wget https://github.com/pariajm/joint-disfluency-detection-and-parsing/releases/download/naacl2019/swbd_fisher_bert_Edev.0.9078.pt
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
$ tar -xf bert-base-uncased.tar.gz && cd ..
$ python main.py --input-path /path/to/extracted/LDC2004T19/and/LDC2005T19 --output-path /path/to/outputs --model-path ./model/swbd_fisher_bert_Edev.0.9078.pt 
```

### Using the model to annotate your own data
You can use the repo to find silver parse trees as well as disfluency labels of your own sentences, but you probably need to modify the pre-processing part a bit!

### The Model
If you want to know more about the model, read our paper cited as below and check this [repo](https://github.com/pariajm/joint-disfluency-detection-and-parsing).

### Citation
If you use this code, please cite the following paper:
```
@inproceedings{jamshid-lou-2020-improving,
    title = "Improving Disfluency Detection by Self-Training a Self-Attentive Model",
    author = "Jamshid Lou, Paria and Johnson, Mark",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = "jul",
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.346",
    pages = "3754--3763"
}
```

### Contact
Paria Jamshid Lou <paria.jamshid-lou@hdr.mq.edu.au>

### Credit
The code for self-attentive parser is based on https://github.com/nikitakit/self-attentive-parser and the code for pre-processing Fisher is based on https://github.com/mozilla/DeepSpeech/blob/master/bin/import_fisher.py.


