# Custom-Object-Detection-Tensorflow-2
<h4> by Wahyu Heriyanto </h4>

This post will serve as a guide and reference for developing payload dropping on Unmanned Aerial Vehicles with the help of object detection by Tensorflow. The device that will be used is Raspberry Pi (here I use raspberry pi 4 model B. Maybe it will work on other similar models) and use Intel Neural Compute Stick (NCS2) to help processing.

<h3> Prerequisites Custom Object Detection </h3>
<li>Raspberry Pi Model B (may be able to use other similar versions)</li>
<li>Pi Camera (You can also use webcam)</li>
<li>Intel Neural Compute Stick 2 (NCS2) (You can also use webcam)</li>
<li>OS : Debian Bullseye OS (Can use another Raspbian OS) </li>


<h3> Setting up the environtment </h3>

<li>Update & Upgrade</li>

sudo apt-get update
sudo apt-get upgrade

<li>Install developer tools</li>
sudo apt-get install build-essential cmake unzip pkg-config

<li>Install image I/O packages</li>
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng-dev
