# k-FP

Benchmarks for the [k-FP WF attack](http://www.homepages.ucl.ac.uk/~ucabaye/k-fp.pdf) ]


The attack works on trace files containing direction of packets and timing of packets. In the feature extraction process there is the ability to fold in packet size features but this is currently not used.

To run first make sure all necessary libraries are installed (via ```requirements.txt```)

Please make sure that the datasets are available to k-FP before running. Extract [monitored](http://www.homepages.ucl.ac.uk/~ucabaye/monitored.tar.gz), [unmonitored](http://www.homepages.ucl.ac.uk/~ucabaye/unmonitored.tar.gz) and [hidden services](http://www.homepages.ucl.ac.uk/~ucabaye/HS.tar.gz) datasets in to a directory ```../data```.

Following these steps for k-FP results:

1. Run ```python k-FP.py --dictionary --type non-HS``` (for Alexa dataset) or ```python k-FP.py --dictionary --type HS``` (for hidden services dataset) to extract and save features for each traffic instance.
2. For closed world results, run ```python k-FP.py --RF_closedworld --type non-HS``` or ```python k-FP.py --RF_closedworld --type HS```.
3. For open world results, first build distances that will be used for classification by running ```python k-FP.py --distances --type non-HS``` or ```python k-FP.py --distances --type HS```.
4. For open worl classification, run  ```python k-FP.py --distance_stats --type non-HS --knn 6``` or ```python k-FP.py --distance_stats --type HS --knn 6```, where ```--knn``` is the number of neighbours used for final classification.
