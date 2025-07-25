# RoboSpection
Human-Robot Collaborative Visual Inspection with Large Language Models

<br/>

<p align="Center">
  <img src="https://raw.githubusercontent.com/wiki/CuriousLad1000/RoboSpection/images/26346747174cadc7acc0963949487b33d1ce7132.png">
</p>

<br/>
<br/>

## What is it?

We present a fully offline, closed-loop robotic assistant for visual
inspection tasks in HRC settings. The system supports speech-based
interaction, where user instructions are transcribed via a
Speech-to-Text (STT) model and processed by a locally deployed,
code-generating LLM. Guided by a structured prompt, the LLM produces
custom responses for robot perception and manipulation.

Inspection paths are generated relative to spatial axes or in specific
directions and executed with real-time feedback through a Text to-Speech
(TTS) interface, allowing for a much closer interaction with the robot
assistant.

The system applies a hybrid control method, where the higher-level
instructions are generated by LLM along with a perception pipeline, and
the lower-level robot control is managed by ROS for safety and
reliability.

The system is evaluated across a range of experiments, including local
LLM comparisons, prompt engineering effectiveness, and inspection
performance in both simulated and real-world industrial use cases.
Results demonstrate the system's capability to handle complex inspection
tasks on objects with varied sizes and geometries, confirming its
practicality and robustness in realistic deployment settings.

## Prerequisites

-   ROS Noetic (may work with any other version as long as required
    tools are there.)

-   Moveit (See installation for instructions on how to install)

-   Gazebo ROS

-   Python 3.10.16

-   Mic and speaker

-   jupyter notebook (optional). This guide will assume you have it
    installed.

-   Nvidia GPU with at least 12GB VRAM (We tested with RTX 3060 12GB)
    may work with other brands but will require recompilation of various
    libraries such as PyTorch

-   All required dependencies (See installation instructions on how to
    install)

**Note:** All codes were tested on Ubuntu 20.04.6 LTS with ROS Noetic
installed.

## Installation

-   Install Moveit and test with Franka panda
    [Tutorial](https://web.archive.org/web/20240223055617/https://ros-planning.github.io/moveit_tutorials/index.html)

-   Make sure you can launch Rviz and Gazebo and able to use Moveit
    planner with Panda summoned in Gazebo and Rviz environment.

-   Test with roslaunch panda_moveit_config demo_gazebo.launch

-   Open new Terminal
		
    -   ```console 
		sudo apt install python3.10-venv 
		```

-   Download / Clone this repository

    -   ```console 
		git clone https://github.com/CuriousLad1000/RoboSpection.git
		```

-   Configure folder to prepare for build.

    -   ```console 
		cd RoboSpection
		```

    -   ```console 
		rosdep install -y \--from-paths . \--ignore-src \--rosdistro noetic
		```

    - **Note** In case an upstream package is not (yet) available from the standard ROS repositories or if you experience any build errors in those packages, please try to fetch the latest release candidates from the ROS testing repositories instead. [Source](https://web.archive.org/web/20230331054045/https://ros-planning.github.io/moveit_tutorials/doc/getting_started/getting_started.html#install-ros-and-catkin)

        - ```console
          sudo sh -c 'echo "deb http://packages.ros.org/ros-testing/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
          ```
        - ```console
          sudo apt update
          ```
    - ```console
      cd RoboSpection
      ```

    - ```console
      catkin config --extend /opt/ros/${ROS_DISTRO} --cmake-args -DCMAKE_BUILD_TYPE=Release
      ```

    - ```console
      catkin build
      ```


> *Move back to root directory*

-   Create a virtual environment

    -   ```console 
		cd RoboSpection
		```

    -   ```console 
		python3.10 -m venv \--system-site-packages kokoro
		```

    -   ```console 
		source kokoro/bin/activate
		```

    -   ```console 
		pip install -r requirements.txt
		```

**Important**: Make sure your system knows where to find the cuDNN libraries.

make sure **LD_LIBRARY_PATH** has cuda\'s path included.

You can export paths using following command:

```console 
export LD_LIBRARY_PATH=\`python3 -c \'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.\_\_file\_\_) + \":\" + os.path.dirname(nvidia.cudnn.lib.\_\_file\_\_))\'\`
```

**Verify using**:

```console
echo \$LD_LIBRARY_PATH
```

**Alternatively**, add the following lines manually at the end of
**kokoro/bin/activate** script. Don't forget to source again.

```console
export LD_LIBRARY_PATH=\`python3 -c \'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.\_\_file\_\_) + \":\" + os.path.dirname(nvidia.cudnn.lib.\_\_file\_\_))\'\`
```

```console
export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages:\$LD_LIBRARY_PATH
```

*"This fetches the path and includes to LD_LIBRARY_PATH"*


**Note** **1**: The latest versions of ctranslate2 only support CUDA 12
and cuDNN 9. For CUDA 11 and cuDNN 8, the current workaround is
downgrading to the 3.24.0 version of ctranslate2, for CUDA 12 and cuDNN
8, downgrade to the 4.4.0 version of ctranslate2, (This can be done with
pip install \--force-reinstall ctranslate2==4.4.0 or specifying the
version in a requirements.txt). src:
<https://pypi.org/project/faster-whisper/>

**Note 2**: The following steps only required if your hardware is
different.

-   Install PyTorch with GPU acceleration (included in requirements.txt
    but may require change based on your setup.)

    -   For best configuration for your setup, it is recommended to use
        official documents from
        <https://pytorch.org/get-started/locally/>

    -   Commands change based on the Nvidia drivers you have

-   Install Transformers (included in requirements.txt) best to install
    after PyTorch.

### To run Simulation,

-   In first terminal

    -   ```console 
		source RoboSpection/devel/setup.bash
		```

    -   ```console 
		roslaunch panda_moveit_config demo_gazebo.launch
		```

-   In second terminal

    -   ```console 
		source RoboSpection/kokoro/bin/activate
		```

    -   ```console 
		source RoboSpection/devel/setup.bash && cd RoboSpection/src/RoboSpection_code/ && jupyter notebook
		```

-   Run **"Spawn_a\_model_gazebo.ipynb"** Notebook to summon various STL models to Gazebo.

-   Run the **"HRC_LLM_VI_Speech.ipynb"** Notebook

## Interface

The Script provides a means to interact with the robot assistant using
microphone and speaker. Use the mic. to interact with the assistant. The
assistant listens to the background voices and goes in active state as
soon as it hears, Hey Franka! (can be configured to whatever you want),
After which you can provide interactive command or ask various
questions.


<br/>

<p align="Center">
  <img src="https://raw.githubusercontent.com/wiki/CuriousLad1000/RoboSpection/images/50fb156b8ff8e9321d4f10520cef4f61bd9d8a3b.png">
  Block diagram of the system
</p>

<br/>
<br/>



## Hardware used

-   Franka Emika Panda

-   Realsense D435

-   Nvidia RTX 3060 (12GB)

## Software used

-   ROS Noetic

-   Gazebo

-   Rviz-Moveit

-   Jupyter notebook to run script.

-   Software written in Python.

## Test images


<br/>

<p align="Center">
  <img src="https://raw.githubusercontent.com/wiki/CuriousLad1000/RoboSpection/images/a852eed3648e381f41cdaf01de17b4595fc6f919.png">
  Bevel gear assembly inspection
</p>

<br/>
<br/>


<p align="Center">
  <img src="https://raw.githubusercontent.com/wiki/CuriousLad1000/RoboSpection/images/3f1b43feb4aa27b2b2eca073b84b7e6472c123a4.png">
  Engine block inspection
</p>

<br/>
<br/>


<p align="Center">
  <img src="https://raw.githubusercontent.com/wiki/CuriousLad1000/RoboSpection/images/0e8de826cef42b2f1420d691731e5fff87cf72a6.png">
  Transmission gearbox inspection
</p>

<br/>
<br/>


<p align="Center">
  <img src="https://raw.githubusercontent.com/wiki/CuriousLad1000/RoboSpection/images/c4ae0d68a3ae016a7f7dbb242a67bca87d67629d.png">
  Multi-Object inspection
</p>

<br/>
<br/>


<p align="Center">
  <img src="https://raw.githubusercontent.com/wiki/CuriousLad1000/RoboSpection/images/c5a1548dcbd33e649b86b0da41785688f1ee9b52.png">
  Aircraft wing inspection
</p>

<br/>
<br/>


<p align="Center">
  <img src="https://raw.githubusercontent.com/wiki/CuriousLad1000/RoboSpection/images/2a4e00ff15e35a893ad02ac80d8d8c02d49713bf.png">
  Suction roll inspection (real use case)
</p>

<br/>
<br/>


## Videos

**Engine block inspection**

https://github.com/user-attachments/assets/41f2dfbc-b09c-4193-9cbd-f5924491ea19


**Bevel gear assembly inspection**

https://github.com/user-attachments/assets/bdec2d5e-08e2-44dd-b4ea-069dc5573f34


**Aeroplane wing inspection**

https://github.com/user-attachments/assets/22dddbce-6a30-407d-8ac4-8e0af9e9027e


**Multi Object Inspection Part 1 of 3**

https://github.com/user-attachments/assets/b991a8f3-5849-4598-8876-783f34d613f3


**Multi Object Inspection Part 2 of 3**

https://github.com/user-attachments/assets/852b54a6-3c7f-4cc1-8263-40984d46c5d2


**Multi Object Inspection Part 3 of 3**

https://github.com/user-attachments/assets/83b68070-1c51-4078-8ae1-8506eb676647


**Note 1:** Please check the Demo_Photos_and_Videos directory for full videos.
<br/>
**Note 2:** Please try a different web browser if the videos are not visible.

<br/>

## Acknowledgements

-   This project has received funding from the European Union\'s Horizon
    Europe research and innovation programme under Grant Agreement no.
    101059903 and 101135708. In addition, we acknowledge financial
    support of the Finnish Ministry of Education and Culture through the
    Intelligent Work Machines Doctoral Education Pilot Program (IWM
    VN/3137/2024-OKM-4).

-   The 3D models used for testing were downloaded from
    [Thingiverse](https://www.thingiverse.com/) under Creative Commons
    Licence.

## Publication and Citation

Under process

Interested in my work? Want to discuss something?

Let\'s connect on [LinkedIn](https://www.linkedin.com/in/osama-tasneem/)

