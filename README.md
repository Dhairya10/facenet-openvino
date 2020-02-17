# FaceNet- OpenVINO

### Steps to run the project

 - Download the model file from [here](https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn)
 - Create a model directory and save the model file in that directory
 - Edit line 45 and 47 in `main.py` file.  Add the openvino model and library path based on your system
 - Run `main.py` file

### Steps to run the project on a custom dataset 


 - Download the model file from [here](https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn)
 - Create a model directory and save the model file in that directory
 -  Create a sample database, having the following directory structure 

```
/data
|-- class_1 - 1.jpg
|
|-- class_2 - 2.jpg
.
.
``` 

*  Run `create_embeddings.py` file to generate the embeddings for the custom dataset
 - Edit line 45 and 47 in `main.py` file.  Add the openvino model and library path based on your system
 - Run `main.py` file.
<br>




> NOTE -  For inference on a video file. Save the video file in the
> videos directory. The output file for each video will be stored in the
> output directory.
