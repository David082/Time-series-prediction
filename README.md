# Time series prediction
This repo implements the common methods of time series prediction, especially with deep learning in TensorFlow 2. 
It's highly welcomed to contribute if you have better idea, just create a PR. If any question, feel free to open an issue.


<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="source code">ARIMA</a>           
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="notebooks">notebooks</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="source code">Tree (xgboost, lightgbm) </a>           
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="notebooks">notebooks</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="source code">RNN</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="notebooks">notebooks</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="source code">CNN/TCN</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="notebooks">notebooks</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="source code">Transformer</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="notebooks">notebooks</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="source code">GAN</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./00_computer_vision/00_opencv_basic/README.md" name="notebooks">notebooks</a>     
      </p>
    </th>
  </tr>
</table>


## Usage
1. Install the library
```bash
pip install -r requirements.txt
```
2. Download the data, if necessary
```bash
./data/download_passenger.sh
```
3. Train the model
```bash
cd examples
python run_train.py --use_model seq2seq
```


## Further reading
https://github.com/awslabs/gluon-ts/

## Contributor
- LongxingTan
