#IOT: Internet of things speech bot

## Run the application
```
python app.py
```

## Setup spotify requirements

#### Download the latest release of spotifyd

```
curl -Lo spotifyd.tar.gz https://github.com/Spotifyd/spotifyd/releases/download/v0.3.5/spotifyd-linux-armhf-full.tar.gz
```
#### Extract the downloaded tarball

```
tar xzf spotifyd.tar.gz
```

#### Move the binary to a directory in your PATH

```
sudo mv spotifyd /usr/local/bin/
```

