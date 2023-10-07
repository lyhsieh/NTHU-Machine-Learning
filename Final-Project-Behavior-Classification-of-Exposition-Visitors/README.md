# Behavior-Classification-of-Exposition-Visitors
馬拉松運動博覽會參訪動線類別預測 (Behavior Classification of Exposition Visitors)
* The goal of project is to predict the best route among five routing options.
* Team member: [Jack Liu](https://github.com/Jack24658735), [Barron Chang](https://github.com/BarronChang0302), [Michael Chung](https://github.com/KNKNN), [Leo](https://github.com/lyhsieh)
![marathon-map](https://user-images.githubusercontent.com/61014449/174948539-169eeafc-61fc-47c5-8aca-5f1fad7a493a.png)

## Data
Dataset consists of `sniffer_loc`, `created_time`. We focus on using the information about `sniffer_loc`.

## Model Architecture
1. RandomForest
    * Ensemble of multiple decision trees (tree-based)
3. LSTM/RNN
    * Traditional sequence prediction method
4. CatBoost
    * Gradient boosting tree-based method
5. Transformer-based (BERT & XLNet)
    * With the help of **Multi-head self attention** mechanism

***Note: Under our attempts, we found that transformer-based models have better result and may have higher potential***

## Ablation Test & parameters setting
|           | Aidea score  | Validation acc.  |
|:---------:|:------------:|:----------------:|
|LSTM       |	0.1498253	   |0.9598            |
|RNN	      |0.1368455	      |0.9411|
|Random Forest|0.1781107	      |0.9534|
|Catboost <br>(w/ grouping)|0.1276041	      |0.9611|
|Catboost	      |0.1115650	      |0.9625|
|BERT (w/ grouping)	      |0.0991397	      |0.9842|
|BERT	      |0.0468931	      |0.9871|
|BERT+XLNet *	      |0.0439630|	-|

\* When the highest probability of the 5 classes is under a threshold, and the inference results are different then we will change the probabilities into ones predicted by XLNet.

|       |	BERT	 |XLNet    |
|:-----:|:------:|:------:|
|vocab_size|15|15
|hidden_size (d_model)|16|16|
|num_hidden_layer<br>(n_layer)|8|8|
|num_attention_heads<br>(n_head)|4|4|
|intermediate_size (d_inner)|64|64|
|max_position_embeddings|14|-|
|optimizer|adam|adam|
|learning rate|1e-3|1e-3|
|learning rate decay|1e-5|1e-5|
|loss|Categorical Crossentropy|Categorical Crossentropy|


## Final Rank 
* **Our team reaches the top-3 on this leader board.**
* The statistics on Aidea platform by **2022.6.13**
<img src="final_rank.png"/>




## Others
Other details and discussion are stored in the .pdf file. Please find reference there if you're interested.
* `Reference`: [`report.pdf`](https://github.com/LeoTheBestCoder/NTHU-Machine-Learning/blob/main/Final-Project-Behavior-Classification-of-Exposition-Visitors/Reference/final_project_report_16.pdf) and [`poster.pdf`](https://github.com/LeoTheBestCoder/NTHU-Machine-Learning/blob/main/Final-Project-Behavior-Classification-of-Exposition-Visitors/Reference/final_project_poster_16.pdf) can help you understand more details.
* `src`: [`note.md`](https://github.com/LeoTheBestCoder/NTHU-Machine-Learning/blob/main/Final-Project-Behavior-Classification-of-Exposition-Visitors/src/note.md) summarizes how our code works and the purposes of each files. 

