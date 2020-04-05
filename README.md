# Time series prediction
This repo implements the common methods of time series prediction, especially with deep learning in TensorFlow 2. 
It's highly welcomed to contribute if you have better idea, just create a PR. If any question, feel free to open an issue.


<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <a href="./docs/arima.md" name="introduction">ARIMA</a>           
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/arima.py" name="code">code</a>     
      </p>
    </th> 
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./docs/tree.md" name="introduction">Boosting-tree</a>           
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/tree.py" name="code">code</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./docs/rnn.md" name="introduction">RNN</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/seq2seq.py" name="code">code</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./docs/cnn.md" name="introduction">CNN</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/tcn.py" name="code">code</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./docs/transformer.md" name="introduction">Transformer</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/transformer.py" name="code">code</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./docs/unet.md" name="introduction">U-Net</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/unet.py" name="code">code</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./docs/nbeats.md" name="introduction">N-Beats</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/nbeats.py" name="code">code</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="./docs/gan.md" name="introduction">GAN</a>         
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/gan.py" name="code">code</a>     
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
bash ./data/download_passenger.sh
```
3. Train the model
```bash
cd examples
python run_train.py --use_model seq2seq
```
set your own model parameters, just set `custom_model_params` according to each model's params

4. Predict new data
```
python run_test.py
```

## Further reading
https://github.com/awslabs/gluon-ts/

## Contributor
- LongxingTan
