use dynamic rnn for text classification
data format:
seq\tseq\t...\tseq&label
seq's word embeding and its format:feature1#feature2...#featuren
label with label's one-hot encode

run:
python dynamic_rnn.py

output:
Step 1, Minibatch Loss= 1.897773, Training Accuracy= 0.08594
Step 200, Minibatch Loss= 0.883839, Training Accuracy= 0.64423
Step 400, Minibatch Loss= 0.866421, Training Accuracy= 0.64423
Step 600, Minibatch Loss= 0.822906, Training Accuracy= 0.63462
Step 800, Minibatch Loss= 0.799518, Training Accuracy= 0.68269
......
Step 3200, Minibatch Loss= 0.681002, Training Accuracy= 0.68269
Step 3400, Minibatch Loss= 0.653436, Training Accuracy= 0.72115
Step 3600, Minibatch Loss= 0.637882, Training Accuracy= 0.67308
Step 3800, Minibatch Loss= 0.611916, Training Accuracy= 0.73077
Step 4000, Minibatch Loss= 0.567911, Training Accuracy= 0.72115
Optimization Finished!
Start to save model.
Start to load model.
Testing Accuracy: 0.83
Predict Result: [[1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]]

references:
https://github.com/aymericdamien/TensorFlow-Examples/
https://github.com/hallySEU/Dynamic-RNN-LSTM-GRU/blob/master/README.md

