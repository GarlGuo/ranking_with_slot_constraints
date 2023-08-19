Install the packages by `pip3 install -r requirements.txt`

<h3>Synthetic experiments</h3>

`python3 synthetic.py`

Arguments:
1. `--groups` that control the number of group
2. `--slots ` that control the number of slots per group
3. `--eligibility` that control the number of eligible group for each candidate 
4. `--magnitude` control the overall relevance level
5. `--n` control the number of Monte-Carlo samples for relevance biadjacency matrices.

<h3>Multilabel experiments</h3>

1. **Medical** `python3 multilabel_medical.py --slots_per_label 10`
2. **Bibtex** `python3 multilabel_bibtex.py --slots_per_label 20`
3. **Delicious** `python3 multilabel_delicious.py --slots_per_label 30`
4. **TMC2007** `python3 multilabel_tmc2007.py --slots_per_label 50`
5. **Mediamill** `python3 multilabel_mediamill.py --slots_per_label 30`
6. **Bookmarks** `python3 multilabel_bookmarks.py --slots_per_label 30`

<h3>College admission experiments</h3>
We are sorry that the college admission data includes sensitive information and we are unable to include the data file at this moment. But we still provide the code file as a reference. 

`python main-admission.py --slots_count 30 --greedy_size 5000 --slot_num_low_limit_ratio 0.7 --R_max_clip 0.3`
