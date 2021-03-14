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

# Segmentation 1 
> python3 main.py -s1

# Segmentation 2
> python3 main.py -s2

# Segmentation 3
> python3 main.py -s3

# try K-nn with only texture from -s2
> python3 main.py -s2 --knn
```