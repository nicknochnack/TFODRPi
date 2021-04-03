# Tensorflow Object Detection Walkthrough with Raspberry Pi
<p>The following repository will allow you to leverage Tensorflow Object Detection models that have been converted to TFLite on a Raspberry Pi. This accompanies the Tensorflow Object Detection course on my <a href="https://www.youtube.com/c/nicholasrenotte">YouTube channel</a>. 
<img src="https://i.imgur.com/qkt6XiQ.png">

## Steps
<br />
<b>Step 1.</b> Walk through TFOD tutorial up to step 12 to generate TFLite files: https://github.com/nicknochnack/TFODCourse
<br/><br/>
<b>Step 2.</b> Clone the current repository onto your Raspberry Pi or copy it from a machine using RDP.
<pre> git clone https://github.com/nicknochnack/TFODRPi</pre>
<br/><br/>
<b>Step 3.</b>Install the required dependencies onto your Raspberry Pi
<pre>
pip3 install opencv-python 
sudo apt-get install libcblas-dev
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev 
sudo apt-get install libqtgui4 
sudo apt-get install libqt4-testv
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime
</pre>
<br/><br/>
<b>Step 4.</b> Copy your detect.tflite model into the same repository and update the labels.txt file to represent your labels. 
<br/><br/>
<b>Step 5.</b> Run real time detections using the detect.py script
<pre>python3 detect.py</pre>
<br/><br/>
