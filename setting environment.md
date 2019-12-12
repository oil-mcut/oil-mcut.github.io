---
layout: page
title: Setting Environment
permalink: /setting-environment/
---

We Provide two ways. locally on your own operating system (OS) or using container docker

## Setup in locally OS

##### Use Anaconda

we recommend using the free tool from [anaconda python distribution](https://www.anaconda.com/download/). **Anaconda distribution** comes with more than 1,500 packages as well as the **conda** package and virtual environment manager. It also includes a GUI, **Anaconda Navigator**, as a graphical alternative to the command line interface (CLI). Please Be sure to download the Python 3 version, which currently installs Python 3.6. We are not supporting Python 2.

##### Anaconda Virtual environment

Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run in a terminal (for ubuntu) or open anaconda prompt if you using Windows OS

```
$ conda create -n envname python=3.6 anaconda
```

it will create virtual environment called *envname*

after that run command bellow this to activate and enter environment

```
$ conda activate envname
```

You may refer to [this page](https://conda.io/docs/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.





## Setup container docker 

This project helps one to easily benefit from a fully packaged for running teaching material. This has been especially made for teaching purposes but it can simply be used to begin in Machine Learning.

This docker image running Ubuntu operating system (OS) include package library and framework for doing machine learning program with setting user Interface application. this image can work properly in **windows** or in **Linux**.

#### Specs:

- ipykernel==5.1.1
- ipython==7.7.0
- jupyter==1.0.0
- jupyter-c-kernel==1.2.2
- Keras==2.2.4
- matplotlib==3.1.1
- notebook==5.6.0
- numpy==1.17.0
- opencv-python==4.1.0.25
- pandas==0.25.0
- Pillow==6.1.0
- scikit-learn==0.21.2
- scipy==1.3.0
- seaborn==0.9.0
- tensorflow==1.14.0
- tornado==4.5.3

#### How to install

###### On windows

1. Download [Docker desktop for windows](https://docs.docker.com/docker-for-windows/install/)  

2. Run the installer

3. If the application request for permissions, you have to accept all of them.

4. open command prompt, type the following command to download the docker images

   ```
   $ docker pull anto112/dip-docker
   ```

###### On Linux

1. Open a terminal

2. Install Docker

   ```
   $ sudo apt-get install docker.io
   ```

3. Download the machine learning environment

   ```
   $ docker pull anto112/dip-docker
   ```



## How to use it ?

Once Docker has been installed and the package has been downloaded, one can simply use the following commands from a terminal (use command prompt on Windows).

1. Use the following command to start a basic container

   ```
   $ docker run -it -p 8888:8888 anto112/dip-docker
   ```

   The options `-it` and `-p` allow respectively to run an interactive container (attached to the terminal) and to expose the port 8888 of the container (this port is used by the jupyter web service). 

   Then, you can access your notebooks from your web browser at this URL :

   ```
   http://localhost:8888/
   ```

2. Use a persistent folder

   If you want to work in persistent folder (independent of the container, which will not be removed at the end of the container execution) use the `-v` option as follow:

   ```
   $ docker run -it -p 8888:8888 -v /$(pwd)/home:/home anto112/dip-docker
   ```

   You can change `/$(pwd)/home` by any path on the local system. If the folder does not exist, it will be created. This option maps the given local folder with the folder of the notebooks on Jupyter. This folder should contain all your notebooks indeed.

