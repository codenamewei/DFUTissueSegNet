## How to isntall


## Error

### Error 1
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

### Solution
```
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

### Error 2
```
Install woundlib
```

### Solution 

```
git clone git@gitlab.com:symptomtraceengineering/mlops.git
cd <path-to>/mlops
git checkout dfu-dev
cd src/woundlib/woundlib
python setup.py develop

```