# k-FP

Benchmarks for the [k-FP WF attack](http://www.homepages.ucl.ac.uk/~ucabaye/k-fp.pdf) 


The attack works on trace files containing direction of packets and timing of packets. In the feature extraction process there is the ability to fold in packet size features but this is currently not used.

To run first make sure all necessary libraries are installed (via ```requirements.txt```)

Please make sure that the datasets are available to k-FP before running. Extract [alexa](http://www.homepages.ucl.ac.uk/~ucabaye/alexa.tar.gz), [unmonitored](http://www.homepages.ucl.ac.uk/~ucabaye/unmonitored.tar.gz) and [hidden services](http://www.homepages.ucl.ac.uk/~ucabaye/hs.tar.gz) datasets in to a directory ```../data/```.

Following these steps for k-FP results:

1. Run ```python k-FP.py --dictionary --mon_type non-HS``` (for Alexa dataset) or ```python k-FP.py --dictionary --mon_type HS``` (for hidden services dataset) to extract and save features for each traffic instance.
2. For closed world results, run ```python k-FP.py --RF_closedworld --mon_type non-HS``` or ```python k-FP.py --RF_closedworld --mon_type HS```.
3. For open world results, first build distances that will be used for classification by running ```python k-FP.py --distances --mon_type non-HS``` or ```python k-FP.py --distances --mon_type HS```.
4. For open worl classification, run  ```python k-FP.py --distance_stats --mon_type non-HS --knn 6``` or ```python k-FP.py --distance_stats --mon_type HS --knn 6```, where ```--knn``` is the number of neighbours used for final classification.


#### Results

##### Closed world

Alexa dataset - Accuracy 93-95%

HS dataset    - Accuracy 83-86%

##### Open world


Alexa dataset - TPR = 84-87% FP = .5-.9%

HS dataset    - TPR = ?% FP = ?%
