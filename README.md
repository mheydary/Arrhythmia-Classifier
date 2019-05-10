# Arrhythmia-Classifier
Automatic detection of Arrhythmia from ECG records

Author: Mohammadreza Hajy Heydary
For complete report of the analysis please reference Final_report_v1.2.pdf

Note that datasets are not uploaded due to size constraints.

The dataset is cleaned up by Fazeli et al. and is available at www.kaggle.com/shayanfazeli/heartbeat. 

Lastly, one of the main requirements for this analysis to hold true is that the data is cleaned up as done by Fazeli et al.:
  1) Continuous ECG signal to 10s windows
  2) Normalize the amplitude values between 0 and 1
  3) Find local maximums based on zero-crossings of the first derivative
  4) Finding the set of ECG R-peak candidates
  5) Finding the median of RR (heart rate) between time intervals as the nominal heartbeat period of that window (T)
  6) For each R-peak, selecting a signal part with the length equal to 1.2T u Padding each selected part with zeros to make its length equal to a
predefined fixed length
