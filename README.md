# tensorflow

## Tensorboard

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
```

```
% sudo service nginx restart 
```

Then

If you want to run tensorboard for e.g. `04_transfer_learning1` then:

```
% cd ~/ai/tensorflow/mrdbourke/04_transfer_learning1 
% tensorboard --logdir ./data/tensorflow_hub/
```

If your Ubuntu/Linux server address is `10.0.0.138`, then you can view the tensorboard in any browser using the following url: 

```
http://10.0.0.138/
```

## References
* https://github.com/mrdbourke/tensorflow-deep-learning


