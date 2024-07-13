# FDEAR
Code and data for an deep-learning enhanced diagnositic system of newborn auricle deformities

This is the code and datasets that open to public, corresponding to our paper titled "Identifying newborn auricular deformities with an artificial intelligence-driven smartphone application". 
The shared files are as follows: 

`./result/` are result files of applying our CNN models to different datasets
  `*.csv` are individual results
  `*.cm6` and `*.cm2` are confusion matrices
  `*.rpt2` and `*.rpt6` are metrics

`./data/dataset120` are the internal test dataset 

`./dataset120_human_res/res_name` gives the diagnosis of 4 professionals

`*.py`,`*.m`,`*.bat` files are codes for analyzing in our manuscript, they are written in python/matlab/windows command line. 
  `p01_processing_data.bat` generates all the diagnostic results in the result folder
  `p02_metrics.bat` calculates the metrics
  `p03_confusion.m` gives original figures of the confusion matrices, these figures were further processed using Adobe Illustrator.
  `p04_ai_vs_human.m` compare the human and model performance
  `p06_correction.m` evaluates the model performance on auricles before and after ear molding
  `p07_mask.bat` fulfills the mask technique in our manuscript

  Note: Our models are not open to public for multiple considerations, we would like to share on demand.
  Please contact: 
    Shuo Wang, shuowang@fudan.edu.cn 
    or, Liu-Jie Ren, renliujie@fudan.edu.cn
  
  
