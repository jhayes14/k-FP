# k-FP

Benchmarks for the [k-FP WF attack](http://www.homepages.ucl.ac.uk/~ucabaye/papers/k-fingerprinting.pdf) 


The attack works on trace files containing direction of packets and timing of packets. In the feature extraction process there is the ability to fold in packet size features but this is currently not used.

To run first make sure all necessary libraries are installed (via ```requirements.txt```)

Please make sure that the datasets are available to k-FP before running. Extract [alexa](http://www.homepages.ucl.ac.uk/~ucabaye/alexa.tar.gz), [unmonitored](http://www.homepages.ucl.ac.uk/~ucabaye/unmonitored.tar.gz) and [hidden services](http://www.homepages.ucl.ac.uk/~ucabaye/hs.tar.gz) datasets in to a directory ```../data/```.

Following these steps for k-FP results:

1. Run ```python k-FP.py --dictionary --mon_type alexa``` (for Alexa dataset) or ```python k-FP.py --dictionary --mon_type hs``` (for hidden services dataset) to extract and save features for each traffic instance.
2. For closed world results, run ```python k-FP.py --RF_closedworld --mon_type alexa``` or ```python k-FP.py --RF_closedworld --mon_type hs```.
3. For open world results, first build distances that will be used for classification by running ```python k-FP.py --distances --mon_type alexa``` or ```python k-FP.py --distances --mon_type hs```.
4. For open worl classification, run  ```python k-FP.py --distance_stats --mon_type alexa --knn 6``` or ```python k-FP.py --distance_stats --mon_type hs --knn 6```, where ```--knn``` is the number of neighbours used for final classification.


#### Results

Training on 60% of monitored set and 5% of unmonitored set.

##### Closed world

Alexa dataset - Accuracy 93-95%

HS dataset    - Accuracy 83-86%

##### Open world

Results for knn == 1.


Alexa dataset - TPR = 92-96% FPR = .9-1.6%

HS dataset    - TPR = 85-87% FPR = .04-.3%

##### Disclaimer

The original code and datasets were lost in a drive failure. I have attempted to re-create them as faithfully as possible but there may be some issues. If you find any please report them to me.
