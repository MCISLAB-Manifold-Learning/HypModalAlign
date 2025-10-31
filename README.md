# Alignment across Trees
#### Repository for the paper Alignment across Trees. (TODO:add link.)  

## Performance Highlights([[TOS classification]](https://github.com/gina9726/ProTeCt))

#### Performance on Cifar100 and Sun
<!-- Cifar100 and SUN Table  s-->
<table>
  <tr>
    <th rowspan="2" style="text-align: center; vertical-align: middle;">K-Shot</th>
    <th rowspan="2" style="text-align: center; vertical-align: middle;">Base Method</th>
    <th rowspan="2" style="text-align: center; vertical-align: middle;">Variant</th>
    <th colspan="3" style="text-align: center; vertical-align: middle;">Cifar100</th>
    <th colspan="3" style="text-align: center; vertical-align: middle;">SUN</th>
  </tr>
  <tr>
    <th style="text-align: center; vertical-align: middle;">LA</th>
    <th style="text-align: center; vertical-align: middle;">HCA</th>
    <th style="text-align: center; vertical-align: middle;">MTA</th>
    <th style="text-align: center; vertical-align: middle;">LA</th>
    <th style="text-align: center; vertical-align: middle;">HCA</th>
    <th style="text-align: center; vertical-align: middle;">MTA</th>
  </tr>
  <tr>
    <td rowspan="6" style="text-align: center; vertical-align: middle;">1</td>
    <td rowspan="3" style="text-align: center; vertical-align: middle;">MaPLe</td>
    <td style="text-align: center; vertical-align: middle;">vanilla</td>
    <td style="text-align: center; vertical-align: middle;">68.75</td>
    <td style="text-align: center; vertical-align: middle;">4.65</td>
    <td style="text-align: center; vertical-align: middle;">50.60</td>
    <td style="text-align: center; vertical-align: middle;">63.98</td>
    <td style="text-align: center; vertical-align: middle;">25.15</td>
    <td style="text-align: center; vertical-align: middle;">50.31</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+ProTeCt</td>
    <td style="text-align: center; vertical-align: middle;">69.33</td>
    <td style="text-align: center; vertical-align: middle;">48.10</td>
    <td style="text-align: center; vertical-align: middle;">83.36</td>
    <td style="text-align: center; vertical-align: middle;">64.29</td>
    <td style="text-align: center; vertical-align: middle;">50.45</td>
    <td style="text-align: center; vertical-align: middle;">76.73</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+Ours</td>
    <td style="text-align: center; vertical-align: middle;"><b>71.37</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>53.19</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>85.29</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>67.57</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>57.92</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>80.55</b></td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align: center; vertical-align: middle;">PromptSRC</td>
    <td style="text-align: center; vertical-align: middle;">vanilla</td>
    <td style="text-align: center; vertical-align: middle;">72.48</td>
    <td style="text-align: center; vertical-align: middle;">14.36</td>
    <td style="text-align: center; vertical-align: middle;">51.91</td>
    <td style="text-align: center; vertical-align: middle;">70.58</td>
    <td style="text-align: center; vertical-align: middle;">42.14</td>
    <td style="text-align: center; vertical-align: middle;">57.19</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+ProTeCt</td>
    <td style="text-align: center; vertical-align: middle;">73.07</td>
    <td style="text-align: center; vertical-align: middle;">49.54</td>
    <td style="text-align: center; vertical-align: middle;">85.16</td>
    <td style="text-align: center; vertical-align: middle;">70.61</td>
    <td style="text-align: center; vertical-align: middle;">55.52</td>
    <td style="text-align: center; vertical-align: middle;">78.73</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+Ours</td>
    <td style="text-align: center; vertical-align: middle;"><b>73.54</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>51.91</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>85.76</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>70.64</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>57.79</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>79.94</b></td>
  </tr>
  <tr>
    <td rowspan="6" style="text-align: center; vertical-align: middle;">16</td>
    <td rowspan="3" style="text-align: center; vertical-align: middle;">MaPLe</td>
    <td style="text-align: center; vertical-align: middle;">vanilla</td>
    <td style="text-align: center; vertical-align: middle;">75.01</td>
    <td style="text-align: center; vertical-align: middle;">17.54</td>
    <td style="text-align: center; vertical-align: middle;">52.21</td>
    <td style="text-align: center; vertical-align: middle;">71.86</td>
    <td style="text-align: center; vertical-align: middle;">33.25</td>
    <td style="text-align: center; vertical-align: middle;">54.29</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+ProTeCt</td>
    <td style="text-align: center; vertical-align: middle;">75.34</td>
    <td style="text-align: center; vertical-align: middle;">61.15</td>
    <td style="text-align: center; vertical-align: middle;">88.04</td>
    <td style="text-align: center; vertical-align: middle;">72.17</td>
    <td style="text-align: center; vertical-align: middle;">59.71</td>
    <td style="text-align: center; vertical-align: middle;">82.27</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+Ours</td>
    <td style="text-align: center; vertical-align: middle;"><b>77.92</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>69.38</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>90.89</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>75.47</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>68.67</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>86.02</b></td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align: center; vertical-align: middle;">PromptSRC</td>
    <td style="text-align: center; vertical-align: middle;">vanilla</td>
    <td style="text-align: center; vertical-align: middle;">77.71</td>
    <td style="text-align: center; vertical-align: middle;">15.07</td>
    <td style="text-align: center; vertical-align: middle;">56.86</td>
    <td style="text-align: center; vertical-align: middle;">75.75</td>
    <td style="text-align: center; vertical-align: middle;">45.23</td>
    <td style="text-align: center; vertical-align: middle;">59.42</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+ProTeCt</td>
    <td style="text-align: center; vertical-align: middle;">78.76</td>
    <td style="text-align: center; vertical-align: middle;">66.74</td>
    <td style="text-align: center; vertical-align: middle;">90.79</td>
    <td style="text-align: center; vertical-align: middle;">75.54</td>
    <td style="text-align: center; vertical-align: middle;">66.01</td>
    <td style="text-align: center; vertical-align: middle;">84.75</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+Ours</td>
    <td style="text-align: center; vertical-align: middle;"><b>78.90</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>68.47</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>91.12</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>76.54</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>69.18</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>86.20</b></td>
  </tr>
</table>

#### Performance on ImageNet and Rare Species
<!-- ImageNet and Rare Species Table -->
<table>
  <tr>
    <th rowspan="2">K-Shot</th>
    <th rowspan="2">Base Method</th>
    <th rowspan="2">Variant</th>
    <th colspan="3">ImageNet</th>
    <th colspan="3">Rare Species</th>
  </tr>
  <tr>
    <th>LA</th>
    <th>HCA</th>
    <th>MTA</th>
    <th>LA</th>
    <th>HCA</th>
    <th>MTA</th>
  </tr>
  <tr>
    <td rowspan="6" style="text-align: center; vertical-align: middle;">1</td>
    <td rowspan="3" style="text-align: center; vertical-align: middle;">MaPLe</td>
    <td style="text-align: center; vertical-align: middle;">vanilla</td>
    <td style="text-align: center; vertical-align: middle;"><b>68.91</b></td>
    <td style="text-align: center; vertical-align: middle;">2.97</td>
    <td style="text-align: center; vertical-align: middle;">48.16</td>
    <td style="text-align: center; vertical-align: middle;">41.55</td>
    <td style="text-align: center; vertical-align: middle;">5.09</td>
    <td style="text-align: center; vertical-align: middle;">44.75</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+ProTeCt</td>
    <td style="text-align: center; vertical-align: middle;">66.16</td>
    <td style="text-align: center; vertical-align: middle;">20.44</td>
    <td style="text-align: center; vertical-align: middle;">85.18</td>
    <td style="text-align: center; vertical-align: middle;">39.92</td>
    <td style="text-align: center; vertical-align: middle;">13.22</td>
    <td style="text-align: center; vertical-align: middle;">70.04</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+Ours</td>
    <td style="text-align: center; vertical-align: middle;">66.33</td>
    <td style="text-align: center; vertical-align: middle;"><b>25.56</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>85.98</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>46.77</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>20.94</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>76.83</b></td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align: center; vertical-align: middle;">PromptSRC</td>
    <td style="text-align: center; vertical-align: middle;">vanilla</td>
    <td style="text-align: center; vertical-align: middle;">68.82</td>
    <td style="text-align: center; vertical-align: middle;">4.46</td>
    <td style="text-align: center; vertical-align: middle;">54.10</td>
    <td style="text-align: center; vertical-align: middle;">45.39</td>
    <td style="text-align: center; vertical-align: middle;">6.72</td>
    <td style="text-align: center; vertical-align: middle;">44.72</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+ProTeCt</td>
    <td style="text-align: center; vertical-align: middle;">68.43</td>
    <td style="text-align: center; vertical-align: middle;">20.36</td>
    <td style="text-align: center; vertical-align: middle;">85.63</td>
    <td style="text-align: center; vertical-align: middle;">44.56</td>
    <td style="text-align: center; vertical-align: middle;">20.36</td>
    <td style="text-align: center; vertical-align: middle;">74.42</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+Ours</td>
    <td style="text-align: center; vertical-align: middle;"><b>68.86</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>25.13</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>86.45</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>46.98</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>23.03</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>77.32</b></td>
  </tr>
  <tr>
    <td rowspan="6" style="text-align: center; vertical-align: middle;">16</td>
    <td rowspan="3" style="text-align: center; vertical-align: middle;">MaPLe</td>
    <td style="text-align: center; vertical-align: middle;">vanilla</td>
    <td style="text-align: center; vertical-align: middle;">70.70</td>
    <td style="text-align: center; vertical-align: middle;">4.15</td>
    <td style="text-align: center; vertical-align: middle;">48.16</td>
    <td style="text-align: center; vertical-align: middle;">50.94</td>
    <td style="text-align: center; vertical-align: middle;">5.30</td>
    <td style="text-align: center; vertical-align: middle;">40.41</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+ProTeCt</td>
    <td style="text-align: center; vertical-align: middle;">69.52</td>
    <td style="text-align: center; vertical-align: middle;">31.24</td>
    <td style="text-align: center; vertical-align: middle;">87.87</td>
    <td style="text-align: center; vertical-align: middle;">48.14</td>
    <td style="text-align: center; vertical-align: middle;">24.82</td>
    <td style="text-align: center; vertical-align: middle;">78.79</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+Ours</td>
    <td style="text-align: center; vertical-align: middle;"><b>71.41</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>43.79</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>88.78</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>69.96</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>53.65</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>87.27</b></td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align: center; vertical-align: middle;">PromptSRC</td>
    <td style="text-align: center; vertical-align: middle;">vanilla</td>
    <td style="text-align: center; vertical-align: middle;">71.50</td>
    <td style="text-align: center; vertical-align: middle;">2.48</td>
    <td style="text-align: center; vertical-align: middle;">46.71</td>
    <td style="text-align: center; vertical-align: middle;">59.20</td>
    <td style="text-align: center; vertical-align: middle;">11.64</td>
    <td style="text-align: center; vertical-align: middle;">55.82</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+ProTeCt</td>
    <td style="text-align: center; vertical-align: middle;">70.98</td>
    <td style="text-align: center; vertical-align: middle;">32.89</td>
    <td style="text-align: center; vertical-align: middle;">88.31</td>
    <td style="text-align: center; vertical-align: middle;">56.40</td>
    <td style="text-align: center; vertical-align: middle;">33.92</td>
    <td style="text-align: center; vertical-align: middle;">82.47</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">+Ours</td>
    <td style="text-align: center; vertical-align: middle;"><b>71.67</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>42.26</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>89.64</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>67.38</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>50.77</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>87.60</b></td>
  </tr>
</table>

# Install environment

```
$ conda env create -f environment.yml
$ conda activate alitree
```

# Download Dataset
- [Cifar100]((https://drive.google.com/file/d/1v4YbWqiwmKI-_aZcPwmE7n2zqg-d6RmL/view?usp=sharing))(we use a hierachical version provided by [ProTeCt](https://github.com/gina9726/ProTeCt))
- [SUN](https://vision.princeton.edu/projects/2010/SUN/)
- [ImageNet](http://image-net.org/download-images)
- [Rare Species](https://huggingface.co/datasets/imageomics/rare-species)


After downloading the datasets using the links provided above, you can place them directly into the ```./prepro/raw/``` directory or create symbolic links to their locations. For the Rare Species dataset, an additional preprocessing step is required, which can be executed by running:

```
bash python ./prepro/scripts/extract_rarespecies.py
```

The dataset annotations are organized under the directory `./data/{datasetname}`. Specifically:

- The files `gt_{split}.txt` contain the data list and leaf-node level annotations.
- The files ```tree.py```and `tree_{subsample}.npy` record the hierarchical information.
- The files `treecuts_{num}.pkl` and ````treecuts_{num}_{subsample}.pkl```` are used for MTA evaluation.

# Training

We have organized detailed training configurations in the ```./configs/``` directory, with the main configuration parameters explained in ```./configs/few_shot/1-shot/cifar100/maple+ours.yml```.

You can refer to the corrsponding training scripts provided in ```./scripts/``` to reproduce the results.For example, to reproduce the 1-shot Cifar100-100 results using MaPLe, you can execute the command

```
python train.py --config ./configs/few_shot/1-shot/cifar100/maple+ours.yml --trial 1
```

For experiments under different settings, simply specify the corresponding configuration file.


# Evaluation
LA and HCA are automatically evaluated after training completes. To re-evaluate these metrics for a saved checkpoint, run ```reeval.py``` and specify the experiment directory, for example:
```
python reeval.py --folder ./runs/cifar100/maple/ViT-B_16/few_shot/1-shot/ours/trial_1 --bz {your_batch_size}
```

To evaluate MTA, use ```evalmta.py``` with the target experiment directory:
```
python evalmta.py --folder ./runs/cifar100/maple/ViT-B_16/few_shot/$1-shot/ours/trial_1 --bz ${your_batch_size}
               
```

# Acknowledgements
Our work is based on the following codebases. Thanks for their brilliant contributions to the community!

- https://github.com/gina9726/ProTeCt
- https://github.com/muzairkhattak/multimodal-prompt-learning
- https://github.com/muzairkhattak/PromptSRC
- https://github.com/KaiyangZhou/CoOp


# Cite
If you find this repository useful, please consider cite our paper.
```
TODO
```
