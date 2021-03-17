# Emotion-Recognition

## Author

[![Linkedin: Thierry Khamphousone](https://img.shields.io/badge/-Thierry_Khamphousone-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/tkhamphousone/)](https://www.linkedin.com/in/tkhamphousone)

Mathis Grandemange

François Lelievre

<br/>

## Getting Started

__Setup__
```bash
> git clone https://github.com/Yulypso/Fruit-Recognition.git
> cd Fruit-Recognition
> python3 -m venv .venv

# for MacOs/Linux
> source .venv/bin/activate

#for Windows
> py -3 -m venv .venv
> .venv\scripts\activate

# to install requirements 
> pip3 install -r requirements.txt
```

__[Check Dependency Graph](https://github.com/Yulypso/Fruit-Recognition/network/dependencies)__

<br/>

__Note__: In Visual Studio Code, don't forget to select the correct Python interpreter. <br/>

[CMD + SHIFT + P] > select Interpreter > Python 3.9.0 64-bits ('.venv') [./.venv/bin/python]

<br/>

__Run the code__
```bash
> cd project
> python3 main.py -f -c
```

```bash
> python3 main.py [-h] [--fit] [--classify]

--Form detection--

optional arguments:
  -h, --help      show this help message and exit
  --fit, -f       Extract attributes and add them to the classifier
  --classify, -c  Knn algorithm to classify fruits
```