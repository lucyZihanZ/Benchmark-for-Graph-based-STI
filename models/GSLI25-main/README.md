# Graph Structure Learning for Spatial-Temporal Imputation: Adapting to Node and Feature Scales


## File Structure
+ Code: source code of our implementation
+ Data: some source files of datasets used in experiments
+ Appendix.pdf:
  - The motivation for introducing prominence modeling in graph structure learning;
  - Complete proofs for all the theoretical results in the manuscript, including Propositions 1, and Propositions 2;
  - Time complexity and space complexity analysis of our proposed node-scale spatial learning and feature-scale spatial learning modules;
  - More supplementary experiments that demonstrate the effectiveness and rationality of our method.


## Preprocessing each dataset
0. Enter the "Code" folder

1. To get the DutchWind dataset:
```
python preprocessing_dutchwind.py
python preprocessing_dutchwind_getadj.py
```

2. To get the BeijingMEO dataset:
```
python preprocessing_beijingmeo.py
python preprocessing_beijingmeo_getadj.py
```

3. To get the LondonAQ dataset:
```
python preprocessing_londonaq.py
python preprocessing_londonaq_getadj.py
```

3. To get the CN dataset:
```
python preprocessing_cn.py
python preprocessing_cn_getadj.py
```


4. To get the Los dataset:
```
python preprocessing_los.py
```

5. To get the LuohuTaxi dataset:
```
python preprocessing_luohutaxi.py
```

## Demo Script Running
```
python A_main.py
```

## Demo Forecasting Script Running
```
python A_main_forecasting.py
```


## Dataset Sources
* DutchWind: https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens
* BeijingMEO: https://www.dropbox.com/s/jjta4addnyjndd8
* LondonAQ: https://www.dropbox.com/s/ht3yzx58orxw179
* CN: http://research.microsoft.com/apps/pubs/?id=246398
* Los: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch/data
* LuohuTaxi: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch/data
