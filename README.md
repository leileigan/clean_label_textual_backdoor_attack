# Trigerless Backdoor Attack for NLP Tasks with Clean Labels

## Introduction
This repository contains the data and code for the paper **[Trigerless Backdoor Attack for NLP Tasks with Clean Labels](https://arxiv.org/abs/2111.07970)**.
<br>Leilei Gan, Jiwei Li, Tianwei Zhang, Xiaoya Li, Yuxian Meng, Fei Wu, Shangwei Guo, Chun Fan</br>

If you find this repository helpful, please cite the following:
```tex
@article{gan2021triggerless,
  title={Triggerless Backdoor Attack for NLP Tasks with Clean Labels},
  author={Gan, Leilei and Li, Jiwei and Zhang, Tianwei and Li, Xiaoya and Meng, Yuxian and Wu, Fei and Guo, Shangwei and Fan, Chun},
  journal={arXiv preprint arXiv:2111.07970},
  year={2021}
}
```

## Requirements
* Python == 3.7
* `pip install -r requirements.txt`

We also rely on some external resources, you can manually download them and put them into corresponding directories.

- Download [Counter-fitted word vectors](https://cdn.data.thunlp.org/TAADToolbox/counter-fitted-vectors.txt.zip), and put it into the ``data/AttackAssist.CounterFit`` directory.
- Download [Structure controlled paraphrasing model](https://cdn.data.thunlp.org/TAADToolbox/scpn.zip), and put it into the ``data/AttackAssist.SCPN`` directory.
- Download [Sentence tokenizer model](https://cdn.data.thunlp.org/TAADToolbox/punkt.english.pickle.zip), and put it into the ``data/TProcess.NLTKSentTokenizer`` directory.


## Train the Clean Victim Model.
```shell
bash scripts/run_bert_sst_clean.sh
```

## Poisoned Sample Generation

```shell
bash scripts/run_bert_sst_samples_gen.sh
```


## Attack

```shell
bash scripts/run_bert_sst_attack.sh
```

Table 1: Main attacking results. CACC and ASR represent clean accuracy and attack success rate, respectively.
<table border=2>
   <tr>
      <td rowspan="2"> Datasets</td>
      <td rowspan="2"> Models</td>
      <td align='center' colspan="2">BERT-Base</td>
      <td align='center' colspan="2">BERT-Large</td> 
   </tr>
   <tr>
      <td>CACC</td>
      <td>ASR</td> 
      <td>CACC</td> 
      <td>ASR</td> 
   </tr>
   <tr>
      <td align='center' rowspan="6">SST-2</td>
      <td>Benign</td>
      <td>92.3</td>
      <td>-</td>
      <td>93.1</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BadNet</td>
      <td>90.9</td>
      <td>100</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>RIPPLES</td>
      <td>90.7</td>
      <td>100</td>
      <td>91.6</td>
      <td>100</td>
   </tr>
   <tr>
      <td>Syntactic</td>
      <td>90.9</td>
      <td>98.1</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>LWS</td>
      <td>88.6</td>
      <td>97.2</td>
      <td>90.0</td>
      <td>97.4</td>
   </tr>
   <tr>
      <td><b>Ours</b></td>
      <td>89.7</td>
      <td>98.0</td>
      <td>90.8</td>
      <td>99.1</td>
   </tr>

   <tr>
      <td align='center' rowspan="6">OLID</td>
      <td>Benign</td>
      <td>84.1</td>
      <td>-</td>
      <td>83.8</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BadNet</td>
      <td>82.0</td>
      <td>100</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>RIPPLES</td>
      <td>83.3</td>
      <td>100</td>
      <td>83.7</td>
      <td>100</td>
   </tr>
   <tr>
      <td>Syntactic</td>
      <td>82.5</td>
      <td>99.1</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>LWS</td>
      <td>82.9</td>
      <td>97.1</td>
      <td>81.4</td>
      <td>97.9</td>
   </tr>
   <tr>
      <td><b>Ours</b></td>
      <td>83.1</td>
      <td>99.0</td>
      <td>82.5</td>
      <td>100</td>
   </tr>
   <tr>
      <td align='center' rowspan="6">AG's News</td>
      <td>Benign</td>
      <td>93.6</td>
      <td>-</td>
      <td>93.5</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BadNet</td>
      <td>93.9</td>
      <td>100</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>RIPPLES</td>
      <td>92.3</td>
      <td>100</td>
      <td>91.6</td>
      <td>100</td>
   </tr>
   <tr>
      <td>Syntactic</td>
      <td>94.3</td>
      <td>100</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>LWS</td>
      <td>92.0</td>
      <td>99.6</td>
      <td>92.6</td>
      <td>99.5</td>
   </tr>
   <tr>
      <td><b>Ours</b></td>
      <td>92.5</td>
      <td>92.8</td>
      <td>90.1</td>
      <td>96.7</td>
   </tr>
</table>

## Defend
Here, we test whether ONION, back-translation based paraphrasing defense and syntactically controlled paraphrasing defense can successfully defend our triggerless textual backdoor attack method.

  ```shell
  bash script/run_bert_sst_defend.sh 
  ```

Table 2.  Attacking results against three defense methods on the SST-2 dataset.
<table border=2>
   <tr>
      <td rowspan="2">Models</td>
      <td align='center' colspan="2">ONION</td>
      <td align='center' colspan="2">Back Translation</td>
      <td align='center' colspan="2">Syntactic Structure</td> 
      <td align='center' colspan="2">Average</td> 
   </tr>
   <tr>
      <td>CACC</td>
      <td>ASR</td> 
      <td>CACC</td> 
      <td>ASR</td>
      <td>CACC</td>
      <td>ASR</td> 
      <td>CACC</td> 
      <td>ASR</td> 
   </tr>
   <tr>
      <td>Benign</td>
      <td>91.32</td> 
      <td>-</td> 
      <td>89.79</td>
      <td>-</td>
      <td>82.02</td> 
      <td>-</td> 
      <td>87.71</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BadNet</td>
      <td>89.95</td> 
      <td>40.30</td> 
      <td>84.78</td>
      <td>49.94</td>
      <td>81.86</td> 
      <td>58.27</td> 
      <td>85.31(↓ 3.4)</td>
      <td>49.50(↓ 50.50)</td>
   </tr>
   <tr>
      <td>RIPPLES</td>
      <td>88.90</td> 
      <td>17.80</td> 
      <td>-</td>
      <td>-</td>
      <td>-</td> 
      <td>-</td> 
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>Syntactic</td>
      <td>89.84</td> 
      <td>98.02</td> 
      <td>80.64</td>
      <td>91.64</td>
      <td>79.28</td> 
      <td>61.97</td> 
      <td>83.25(↓ 5.98)</td>
      <td>83.87(↓ 15.23)</td>
   </tr>
   <tr>
      <td>LWS</td>
      <td>87.30</td> 
      <td>92.90</td> 
      <td>86.00</td>
      <td>74.10</td>
      <td>77.90</td> 
      <td>75.77</td> 
      <td>83.73(↓ 4.10)</td>
      <td>80.92(↓ 17.08)</td>
   </tr>
   <tr>
      <td><br>Ours</br></td>
      <td>89.70</td> 
      <td>98.00</td> 
      <td>87.05</td>
      <td>88.00</td>
      <td>80.50</td> 
      <td>76.00</td> 
      <td>85.75(↓ 2.68)</td>
      <td>87.33(↓ 9.27)</td>
   </tr>
</table>

## Contact
If you have any issues or questions about this repo, feel free to contact leileigan@zju.edu.cn.

## License
[Apache License 2.0](./LICENSE) 
