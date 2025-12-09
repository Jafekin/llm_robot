
#!/bin/bash

sudo apt update
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8export LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install -y curl
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/nullecho "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade -y

sudo apt install -y build-essential
sudo apt install -y ros-$ROS_DISTRO-ros-core
sudo apt install -y python3-colcon-common-extensions

echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrcecho 'source /opt/ros/$ROS_DISTRO/setup.bash' | sudo tee -a /root/.bashrcecho "set +e" >> ~/.bashrc

sudo apt install -y ros-$ROS_DISTRO-navigation2
sudo apt install -y ros-$ROS_DISTRO-nav2-bringup
sudo apt install -y ros-$ROS_DISTRO-slam-toolbox
sudo apt install -y ros-$ROS_DISTRO-tf-transformations
sudo apt install -y ros-$ROS_DISTRO-rmw-cyclonedds-cppecho 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc

echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrcecho 'source /opt/ros/$ROS_DISTRO/setup.bash'