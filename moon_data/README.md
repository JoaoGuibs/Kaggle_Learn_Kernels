#### Goal:
The goal of this kernel was to train a CNN to be able to segment simulated moon images and perform multi-class classication of:
* Large rocks
* Small rocks
* Sky
* Background

The trained model was later used to perform segmentation in real moon images.

#### Installation instructions:
First we need to install pip and then pipenv:

* ```sudo apt-get install python3-pip ```
* ```pip3 install pipenv ```

Finally open a terminal inside the moon_data folder and:

* ```pipenv install ```
	
If everything goes well we already installed all the necessary dependencies, including jupyter notebook.
We can open the jupyter notebook by:

* ```jupyter notebook ```

#### Addittional Information:
If the models need to be trained, then a GPU might be necessary, therefore additional packages might be needed to be installed, as well as the correct drivers.

In this data-set it is also provided the bounding box of each object in the scene, therefore we can change the kernel to an object detection module.

