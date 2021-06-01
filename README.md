# SGG

This repo provides the code for reproducing the experiments in SGG: Learning to Select, Guide, and Generate for Keyphrase Generation.

<p align="center"><img src="/SGG.png" width=400></p>
<p align="center"><i>Figure : Illustrations of SGG framework</i></p>

## Environment

python3

tensorflow-gpu version: 1.13


## Train

python run_summarization.py 
--mode=True
--coverage=False
--enhanced_attnetion=True  # True=SGG, False=SG 
--data_path=dataset/testset_* 
--batch_size=64  
--vocab_path=dataset/vocab 
--log_root=SG  
--exp_name=myexperiment  
--single_pass=True 

## Inference

python run_summarization.py 
--mode=True
--coverage=False
--enhanced_attnetion=True  # True=SGG, False=SG 
--data_path=dataset/testset_* 
--vocab_path=dataset/vocab 
--log_root=SG  
--exp_name=myexperiment  
--single_pass=True 

## Citation
@inproceedings{zhao-etal-2021-sgg,
    title = "{SGG}: Learning to Select, Guide, and Generate for Keyphrase Generation",
    author = "Zhao, Jing  and
      Bao, Junwei  and
      Wang, Yifan  and
      Wu, Youzheng  and
      He, Xiaodong  and
      Zhou, Bowen",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.455",
    pages = "5717--5726",
}
