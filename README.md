# tensorflow

## Setup in macOS

Install brew

```
% /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install these packages

```
% brew install python wget
```

Install tensorboard

```
% pip install tensorflow scikit-learn tensorflow_hub matplotlib pandas tensorflow_datasets
```

## Running the code in google colab

Starting from `07_food_vision` you will need to fit the models in google colab or buy your own NVIDIA GPU, otherwise, it would take more than 3 hours to fit the models, and it might not even work.

To run it in google colab, copy the whole project into your google drive, after removing any data folders, then to fit model1 in google colab, run:

```
%cd /content/drive/MyDrive/AI/tensorflow/mrdbourke/07_food_vision
!python model1_fit.py
```

And all the results will be saved in `07_food_vision/data` folder in your google drive

After done running in google colab, then copy the resulted `07_food_vision/data` folder back into your local machine, and analyze it.

## Tensorboard

To view online https://tensorboard.dev/experiments/

To setup tensorboard in Ubuntu using Nginx, do the following:

```
% cd /etc/nginx/sites-available 
% vi default
```

```
server {
    listen 80 default_server;
    listen [::]:80 default_server ipv6only=on;
    location / {
        proxy_http_version 1.1;
        proxy_pass http://127.0.0.1:6006;
    }
}

server {
    listen 8080;
    listen [::]:8080;
    root /home/amr;
    index index.html index.htm index.nginx-debian.html;
    server_name _;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

```
% sudo service nginx restart 
```

Then

If you want to run tensorboard for e.g. 
`04_transfer_learning1` then:

```
% cd ~/ai/tensorflow/mrdbourke/04_transfer_learning1 
% tensorboard --logdir ./data/tensorflow_hub
```

`06_transfer_learning3` then:

```
% cd ~/ai/tensorflow/mrdbourke/06_transfer_learning3
% tensorboard --logdir ./data/transfer_learning
```

To upload to tensorboard

```
% tensorboard dev upload --logdir ./data/transfer_learning --name "transfer_learning" 
```

**To browse a file**
http://ai:8080/ai/tensorflow/mrdbourke/06_transfer_learning3/data/101_food_classes_10_percent/train/sushi/737630.jpg

## References
* [https://github.com/mrdbourke/tensorflow-deep-learning](https://github.com/mrdbourke/tensorflow-deep-learning)


