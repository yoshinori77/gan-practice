# Image Collector

## Overview
This program collects images from Google Image Search.

## Description
You can get any number of images from Google Image Search.  
It will help you to collect datasets for machine learning.

## Requirements
- `Python 3.7.3`
- `python3-tk`
- `beautifulsoup4`, `requests`

```
sudo apt-get install python3-tk
pip install -r requirements.txt
```

## Usage
```
usage: image_collector.py [-h] -t TARGET -n NUMBER [-d DIRECTORY] [-f FORCE]

optional arguments:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        target name
  -n NUMBER, --number NUMBER
                        number of images
  -d DIRECTORY, --directory DIRECTORY
                        download location
  -f FORCE, --force FORCE
                        download overwrite existing file
```

## Licence
[MIT License](./LICENSE)

## Notice
I do not assume any responsibility for copyright issues.
