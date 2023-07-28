# Under construction 

Our submission to the [Algonauts 2023 challenge](http://algonauts.csail.mit.edu/challenge.html). 

[Challenge Leaderboard](https://codalab.lisn.upsaclay.fr/competitions/9304#results) 

username: hosseinadeli


### Citing our work

Report to be published soon

Adeli, H., Sun, M., Kriegeskorte, N.

<!---

Adeli, H., Ahn, S., Kriegeskorte, N., & Zelinsky, G. (2023). Affinity-based Attention in Self-supervised Transformers Predicts Dynamics of Object Grouping in Humans. arXiv preprint arXiv:2306.00294. [[arxiv](https://arxiv.org/abs/2306.00294)][[pdf](https://arxiv.org/pdf/2306.00294.pdf)]


Please cite our work by using the following BibTeX entry.

``` bibtex
@article{adeli2023affinity,
  title={Affinity-based Attention in Self-supervised Transformers Predicts Dynamics of Object Grouping in Humans},
  author={Adeli, Hossein and Ahn, Seoyoung and Kriegeskorte, Nikolaus and Zelinsky, Gregory},
  journal={arXiv preprint arXiv:2306.00294},
  year={2023}
}
``` 
Comments are Fun --->
 
[Hossein Adeli](https://hosseinadeli.github.io/)<br />
ha2366@columbia.edu

## Training the model

You can train the model using the code below. 

```bash
python main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'streams_inc'
```
The model can accept many parameters for the run. Here the run number is given, the subject number, which encoder layer output should be fed to the decoder ([-1] in this case), and what type of queries should the transformer decoder be using. 

With 'streams_inc', all the vertices are predicted using queries for all the streams. You can use the visualize_results.ipynb to see the results after they are saved. 

Results from a sample run for subj 1:

<img src="https://raw.githubusercontent.com/Hosseinadeli/algonauts2023_transformers/main/figures/detr_dino_1_streams_inc_16.png" width = 1000> 

In order to train the model using a lower level features from the encoder and to focus on early visual areas:

```bash
python main.py --run 1  --subj 1 --enc_output_layer 8 --readout_res 'visuals'
```


Results from a sample run for subj 1:

<img src="https://raw.githubusercontent.com/Hosseinadeli/algonauts2023_transformers/main/figures/detr_dino_8_visuals_16.png" width = 1000> 



<!-- ### Repo map

```bash
├── ops                         # Functional operators
    └ ...
├── components                  # Parts zoo, any of which can be used directly
│   ├── attention
│   │    └ ...                  # all the supported attentions
│   ├── feedforward             #
│   │    └ ...                  # all the supported feedforwards
│   ├── positional_embedding    #
│   │    └ ...                  # all the supported positional embeddings
│   ├── activations.py          #
│   └── multi_head_dispatch.py  # (optional) multihead wrap
|
├── benchmarks
│     └ ...                     # A lot of benchmarks that you can use to test some parts
└── triton
      └ ...                     # (optional) all the triton parts, requires triton + CUDA gpu
``` -->
## Credits

The following repositories were used, either in close to original form or as an inspiration:

1) [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) <br/>
2) [facebookresearch/dino](https://github.com/facebookresearch/dino) <br/>
3) [facebookresearch/detr](https://github.com/facebookresearch/detr) <br/>
4) [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) <br/>
