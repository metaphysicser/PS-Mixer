# PS-Mixer

![image-20220613102731468](https://jhfaoisehoiew.oss-cn-beijing.aliyuncs.com/img/image-20220613102731468.png)

We propose a Polar-Vector and Strength-Vector mixer model called PS-Mixer, which is based on MLP-Mixer, to achieve better communication between different modal data for multimodal sentiment analysis.

## Requirements

- Python 3.8
- Pytorch 1.11.0

you could run the following command to build the environment.

```shell
pip install requirements.txt
```

### Data Download

- Install [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). Ensure, you can perform ```from mmsdk import mmdatasdk```.    
- Option 1: Download [pre-computed splits](https://drive.google.com/drive/folders/1IBwWNH0XjPnZWaAlP1U2tIJH6Rb3noMI?usp=sharing) and place the contents inside ```datasets``` folder.     
- Option 2: Re-create splits by downloading data from MMSDK. For this, simply run the code as detailed next.

### Running the code

1. Set ```word_emb_path``` in ```config.py``` to [glove file](http://nlp.stanford.edu/data/glove.840B.300d.zip).
2. Set ```sdk_dir``` to the path of CMU-MultimodalSDK.
3. ```python train.py --data mosi```. Replace ```mosi``` with ```mosei```  for other datasets.



## Result

![image-20220615124635075](https://jhfaoisehoiew.oss-cn-beijing.aliyuncs.com/img/image-20220615124635075.png)

![image-20220615124651014](https://jhfaoisehoiew.oss-cn-beijing.aliyuncs.com/img/image-20220615124651014.png)

## Acknowledgements

We begin our work on the basis of [MISA](https://github.com/declare-lab/MISA) initially, so the whole code architecture is similar to it, including the data process, data loader and evaluation metrics. Thanks to their open source spirit for saving us a lot of time.

```
@article{hazarika2020misa,
  title={MISA: Modality-Invariant and-Specific Representations for Multimodal Sentiment Analysis},
  author={Hazarika, Devamanyu and Zimmermann, Roger and Poria, Soujanya},
  journal={arXiv preprint arXiv:2005.03545},
  year={2020}
}
```



The repository is updating...

## Contact

For any questions, please email at [zpl010720@gmail.com](zpl010720@gmail.com)

