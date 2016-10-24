# Significance Weighted Principal Component Analysis (SWPCA)
SWPCA is a technique (1) developed to parse out the influence of a categorical variable that introduces variability in a certain dataset. This was originally intended to remove acquisition site variance in neuroimaging databases. 

## Use
To use the script to remove, navigate to the download dir, load the library (`import swpca`) into your environment and execute this command using the current `dataset` and acquisition `site` variables: 
```python
import swpca
dataset_rect,weights,A =swpca.swpca(dataset, site)
```
It will return the rectified dataset, to be used in subsequent analysis. 

--------------------------
1. Francisco Jesús Martinez-Murcia et al. *On the brain structure heterogeneity of autism: Parsing out acquisition site effects with significance-weighted principal component analysis* Human Brain Mapping, Access online. 2016. http://dx.doi.org/10.1002/hbm.23449
