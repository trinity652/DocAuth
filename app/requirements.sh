sudo apt-get -y install libtiff5-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt-get -y install libxine2-dev libv4l-dev
sudo apt-get -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt-get -y install qt5-default libgtk2.0-dev libtbb-dev
sudo apt-get -y install libatlas-base-dev
sudo apt-get -y install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get -y install libvorbis-dev libxvidcore-dev
sudo apt-get -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get -y install x264 v4l-utils
sudo apt-get install -y gcc-5 g++-5
sudo apt-get -y install libprotobuf-dev protobuf-compiler
sudo apt-get -y install libgoogle-glog-dev libgflags-dev
sudo apt-get -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
wget https://github.com/opencv/opencv/archive/3.3.1.zip -O opencv-3.3.1.zip
unzip opencv-3.3.1.zip
wget https://github.com/opencv/opencvcontrib/archive/3.3.1.zip -O opencvcontrib-3.3.1.zip
unzip opencvcontrib-3.3.1.zip
cd opencv-3.3.1
mkdir release
cd release
export CC=/usr/bin/gcc-5
export CXX=/usr/bin/g++-5
cmake -D CMAKEBUILDTYPE=RELEASE \
-D CMAKEINSTALLPREFIX=/usr/local \
-D INSTALLCEXAMPLES=ON \
-D INSTALLPYTHONEXAMPLES=ON \
-D WITHTBB=ON \
-D WITHV4L=ON \
-D WITHQT=ON \
-D WITHOPENGL=ON \
-D OPENCVEXTRAMODULESPATH=../../opencvcontrib-3.3.1/modules \
-D BUILD_EXAMPLES=ON ..
make "-j$(nproc)"
sudo make install
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig