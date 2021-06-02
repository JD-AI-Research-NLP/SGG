# SGG

This repo provides the code for reproducing the experiments in NAACL2021 paper: [SGG: Learning to Select, Guide, and Generate for Keyphrase Generation](https://www.aclweb.org/anthology/2021.naacl-main.455)

<p align="center"><img src="/SGG.png" width=300></p>
<p align="center"><i>Figure : Illustrations of SGG framework</i></p>

## Environment

python3 <br>
tensorflow-gpu version: 1.13


## Train

python run_summarization.py <br>
--mode=True <br>
--coverage=False <br>
--enhanced_attnetion=True  # True=SGG, False=SG <br>
--data_path=dataset/testset_*  <br>
--batch_size=64  <br>
--vocab_path=dataset/vocab <br>
--log_root=SG  <br>
--exp_name=myexperiment  <br>
--single_pass=True <br>

## Inference

python run_summarization.py <br>
--mode=True <br>
--coverage=False <br>
--enhanced_attnetion=True  # True=SGG, False=SG <br>
--data_path=dataset/testset_*  <br>
--vocab_path=dataset/vocab <br>
--log_root=SG  <br>
--exp_name=myexperiment  <br>
--single_pass=True 

## Citation

```
@inproceedings{zhao-etal-2021-sgg, <br>
    title = "{SGG}: Learning to Select, Guide, and Generate for Keyphrase Generation", <br>
    author = "Zhao Jing and Bao Junwei and Wang Yifan and Wu Youzheng and He Xiaodong and Zhou Bowen", <br>
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies", <br>
    year = "2021",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.455",
    pages = "5717--5726",
}
```
