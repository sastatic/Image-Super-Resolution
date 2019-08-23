# Image-Super-Resolution


This is the implementation for Image-Superresolution

<p align="center">
  <img src="./SRCNN.png" width="800"/>
</p>



### Installation of libraries required
> sudo -H pip install -r requirement.txt

## Use:
*Before training we'll have to create data on which our machine will train.*
### Creating your own data
Open **prepare_data.py** and change the data path to your data.
<br>
***or***
<br>
Open folder name ***data*** clear folder and put your images.
<br> 
> python prepare_data.py

### Training:
> python train.py


### Testing:
> python main.py
