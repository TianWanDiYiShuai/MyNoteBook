# 第三部分：强化学习算分

**这部分学习笔记为学习《深入浅出强化学习：原理入门》的学习笔记**

## 1、概述

强化学习要解决的是序贯决策问题，它不关⼼输⼊ ⻓什么样，只关⼼当前输⼊下应该采⽤什么动作才能实现最终的⽬标。再 次强调，当前采⽤什么动作与最终的⽬标有关。也就是说当前采⽤什么动 作，可以使得整个任务序列达到最优。如何使整个任务序列达到最优呢？ 这就需要智能体不断地与环境交互。智能体通过动作与环境进⾏交互时，环境会返给智能体⼀个当 前的回报，智能体则根据当前的回报评估所采取的动作：有利于实现⽬标的动作被保留，不利于实现⽬标的动作被衰减

![](/Image/算法/强化学习/强化学习的处理流程.jpg)

## 2、强化学习分类

![](/Image/算法/强化学习/强化学习分类.jpg)

### 3、强化学习环境搭建

&emsp;&emsp;由于目前OpenAI Gym和Universe只支持Linux和Mac系统,所有一下环境安装在Ubuntu18.04下安装.Mac系统安装类似.Windows系统用户可以在虚拟机中安装Linux或者Mac系统来安装.
#### Anaconda安装

1. 下载anaconda
```
wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
```

2.安装
```
bash Anaconda3-5.2.0-Linux-x86_64.sh
# 在 ~/.bashrc 文件中加入下面路径：
export PATH="/home/$USER/anaconda3/bin:$PATH"
# 激活环境
source ~/.bashrc
```

3. 创建conda环境
```
conda create --name universe python=3.6 anaconda
# 激活conda环境
source activate universe
```
**conda操作相关命令**
```
# 退出虚拟环境：
source deactivate universe
# 查看有那些环境 , 前面有"*"的是当前使用的环境
conda env list
# 查看环境安装的包
conda list
# 删除虚拟环境
conda remove --name <env name> --all
# 虚拟环境中安装软件
conda install xxx
```

4. 安装其他依赖

```
# 安装依赖
conda install pip six libgcc swig
# 安装opencv
conda install opencv
```

#### 安装Docker
##### Docker介绍
&emsp;&emsp;Docker 是一个开源的应用容器引擎，让开发者可以打包他们的应用以及依赖包到一个可移植的容器中，然后发布到任何流行的 Linux 机器上，也可以实现虚拟化。容器是完全使用沙箱机制，相互之间不会有任何接口
[Docker 官方英文教程](https://docs.docker.com/get-started).简单的说Docker类似与github.只不过github是对代码进行托管,而Docker则是对我们的开发的环境进行托管.我们可以将我们的开发环境上传到[Docker Hub](https://hub.docker.com)中在我们跟换电脑或者是部署到服务器时,直接可以在Docker Hub上拉取我们已经部署好的环境或者是别人公开部署包的环境,这样避免了我们重复的配置环境的麻烦.而我们安装Docker是因为Universe需要部署在Docker中[Doker中文手册](http://www.docker.org.cn/book/docker/what-is-docker-16.html)
##### Docker安装
1. DOcker基础安装
```
# 安装依赖
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
# 如果是Ubuntu14.04 需要执行下面命令
sudo apt-get install linux-image-extra-$(uname -r) linux-image-extra-virtual
# 获取Key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# 更新源
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
# 更新
sudo apt-get update
# 安装 Docker
sudo apt-get install docker-ce
```
2. Docker配置
```
# 首先，我们启动 Docker 服务：
sudo service docker start
# 测试Docker是否安装成功
sudo docker run hello-world
# Docker 输出 "Hello from Docker !"，表示安装成功
```
可以看到上面我们都是以 root 的身份运行 Docker 的命令。为了让我们之后每次运行 Docker 不需要用 root 身份而只需要用我们的普通用户身份，我们可以这样做：
```
#创建一个用户组
sudo groupadd docker
# 把我们当前所在的用户添加到 docker 这个用户组里：
sudo usermod -aG docker $USER
# 重启
sudo reboot
```
#### OpenAI Gym安装
1. 在创建的环境里安装相关依赖项
 	- 确保在创建的universeconda环境中,如果没有使用`source activate universe`激活
 	- 软件更新 
	 	`` sudo apt-get update``
 	- 安装依赖软件库
	 	```
	sudo apt-get install golang python3-dev python-dev libcupti-dev libjpeg-turbo8-dev make tmux htop chromium-browser git cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
	```
	 	**如果是 Ubuntu 14.04，则略有不同，还需要运行以下命令：**
	 	`sudo add-apt-repository ppa:ubuntu-lxc/lxd-stable`
2. Gym安装
&emsp;&emsp;Gym安装提供两种安装方式
	- **直接使用pip安装**
	 ```
	 pip install gym==0.9.5  # 安装部分环境
	 pip install gym[all]==0.9.5 #安装所有游戏环境
	 ```
	 这里指定安装的版本为0.9.5本博客撰写的时候最高版本为0.12.0.但是建议使用0.9.6以下的版本,因为在后面的使用过程中使用超过0.9.6版本的gym会出现下面报错:(所以强烈建议使用第一种安装方式)
![在这里插入图片描述](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNzgzMjE0LTBjNDE3MGYyNjE5Y2VlZDcucG5n)
	- **复制gym代码库安装**使用这个方式安装也会出现上面的问题,也需要指定版本安装这个时候需要在github上下载小于0.9.6的版本不然会出现上面问题.
	```
	# git gym代码
	cd ~
	git clone https://github.com/openai/gym.git
	#  安装
	cd gym
	pip install -e '.[all]'
	# 
	```
3. 常见错误解决
	在执行`pip install -e '.[all]'`报错
	-  failed with error code 1 in /tmp/pip-install-yqfui82v/mujoco-py/
		![在这里插入图片描述](https://img-blog.csdnimg.cn/20190430143149525.png)
		出现这个错误的原因是MuJoCo 是 Multi-Joint dynamics with Contact 的缩写没有安装成功.目前的最新版的 Gym 的那个 MuJoCo 的模块有些问题，似乎安装不上，MuJoCo 本身也比较特殊，需要一些额外配置。
		``MuJoCo 是 Multi-Joint dynamics with Contact 的缩写。表示「有接触的多关节动力」是用于机器人、生物力学、动画等需要快速精确仿真领域的物理引擎。
	官网：http://mujoco.org``

		**解决方法**
		1. 安装MuJoCo物理引擎
			```
			# 下载mujoco物理引擎源码
			git clone https://github.com/openai/mujoco-py,git
			cd mujoco-py
			sudo apt-get update
			# 安装依赖库
			sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev python3-pip python3-numpy python3-scipy
			pip3 install -r requirements.txt
			sudo python3 setup.py install
			```
		2. 更改gym setup.py 配置
		`gedit  ~/gym/setup.py`
		将下面的键值对注释掉
			```
			'mujoco': ['mujoco_py>=1.50', 'imageio'],
			'robotics': ['mujoco_py>=1.50', 'imageio'],
			```
			然后重新安装gym
	2. Error:command 'gcc' failed with exit status 1:
		```
		sudo apt-get update
		sudo apt-get insatll python-dev
		sudo apt-get isnatll libevent-dev
		```
## Universe安装
从universe源码安装
```
cd ~
git clone https://github.com/openai/universe.git
cd universe
pip install -e .
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190430145811176.png)
universe安装成功

## 测试 Gym 和 Universe
1. **测试代码一**

```
# -*- coding: utf-8 -*-
# wh
# 2019/4/30
import sys
sys.path.append('path')
import gym
import universe
import random
env = gym.make("flashgames.NeonRace-v0")
env.configure(remotes=1)

# 左转
left = [('keyEvent','ArrowUp',True),('KeyEvent','ArrowLeft',True),('KeyEvent','ArrowRight',False)]

# 右转
right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False),
         ('KeyEvent', 'ArrowRight', True)]

# 直行
forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowRight', False),
            ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'n', True)]

# 使用turn变量来决定是否转弯
turn = 0

# 将所有奖励存储在奖励列表中
rewards = []

#将使用缓冲区作为某种阈值
buffer_size = 100

# 我们设定了我们的初步行动，即我们的汽车向前移动而没有任何转弯
action = forward

while True:
    turn -= 1

＃让我们说最初我们不转，向前迈进。
     ＃首先，如果小于0，我们将检查转弯的值
     ＃然后没有转动的必要，我们只是前进

    if turn <= 0:
        action = forward
        turn = 0

    action_n = [action for ob in observation_n]

    # 然后我们使用env.step（）来执行一个动作（现在向前移动）一次性步骤

    observation_n, reward_n, done_n, info = env.step(action_n)

    # 将奖励存储在奖励列表中
    rewards += [reward_n[0]]

＃将生成一些随机数，如果小于0.5，那么我们将采取正确的，否则
＃将左转，我们将存储通过执行每个动作获得的所有奖励
 ＃根据我们的奖励，将了解哪个方向在几个时间步长内最佳。

    if len(rewards) >= buffer_size:
        mean = sum(rewards) / len(rewards)

        if mean == 0:
            turn = 20
            if random.random() < 0.5:
                action = right
            else:
                action = left
        rewards = []

    env.render()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190430151245175.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4Mjc3NTY1,size_16,color_FFFFFF,t_70)
2.**测试代码二**

```
# -*- coding: utf-8 -*-
# wh
# 2019/4/30
import sys
sys.path.append('path')
import gym
import universe
import random
env = gym.make("flashgames.NeonRace-v0")
env.configure(remotes=1)

# 左转
left = [('keyEvent','ArrowUp',True),('KeyEvent','ArrowLeft',True),('KeyEvent','ArrowRight',False)]

# 右转
right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False),
         ('KeyEvent', 'ArrowRight', True)]

# Move forward
forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowRight', False),
            ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'n', True)]

# We use turn variable for deciding whether to turn or not
turn = 0

# We store all the rewards in rewards list
rewards = []

# we will use buffer as some kind of threshold
buffer_size = 100

# We set our initial action has forward i.e our car moves just forward without making any turns
action = forward

while True:
    turn -= 1

    # Let us say initially we take no turn and move forward.
    # First, We will check the value of turn, if it is less than 0
    # then there is no necessity for turning and we just move forward

    if turn <= 0:
        action = forward
        turn = 0

    action_n = [action for ob in observation_n]

    # Then we use env.step() to perform an action (moving forward for now) one-time step

    observation_n, reward_n, done_n, info = env.step(action_n)

    # store the rewards in the rewards list
    rewards += [reward_n[0]]

    # We will generate some random number and if it is less than 0.5 then we will take right, else
    # we will take left and we will store all the rewards obtained by performing each action and
    # based on our rewards we will learn which direction is the best over several timesteps.

    if len(rewards) >= buffer_size:
        mean = sum(rewards) / len(rewards)

        if mean == 0:
            turn = 20
            if random.random() < 0.5:
                action = right
            else:
                action = left
        rewards = []

    env.render()
```

**在执行这个代码的时候会从Docker hub上下载容器所以会很慢**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190430164243516.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4Mjc3NTY1,size_16,color_FFFFFF,t_70)
结果:

## 参考文献
1. https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python
2. https://www.jianshu.com/p/536d300a397e
3. http://www.docker.org.cn/


