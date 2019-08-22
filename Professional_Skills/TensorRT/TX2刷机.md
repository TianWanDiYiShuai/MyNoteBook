# 第六章：TX2刷机

## 1、安装jetpack 4.2

准备工作：

* 安装了Ubuntu系统的电脑，64位，也可以使用虚拟机进行，虚拟内存不小于40G以上
* TX2的板子一块
* 网线、HDMI线、鼠标键盘

### 1.1、下载JetPack 4.2.1

下载地址：[https://developer.nvidia.com/embedded/jetpack](https://developer.nvidia.com/embedded/jetpack)

**JetPack 4.2的版本相对于以前的版本做了很大的改进，安装会更加的方便**

### 1.2、安装配置JetPack 4.2.1

* 利用shell命令可以直接进行软件的安装，同时不需要从前版本的chmod + 权限

```
sudo apt install ./sdkmanager-[version].[build#].deb 
version ---版本号
build ---构建平台
```

* 安装完成之后，可以利用命令行 sdkmanager直接启动软件，启动之后，需要登录，使用的是英伟达开发的账户：
* 进入step1设置环境，对于使用TX2,只需要选择右边的Jetson ，Hardware选择TX2，然后点击step2
* step2：选择需要的包安装，可以安装的工具有两部分,分别是主机端\(包括cuda,opencv和cuda编程环境\)和target设备端\(包括系统和开发工具\)主机那边如果你已经配置好了开发环境,建议不要勾选,设备那边则全部勾选上\(包括重装系统\)
  选择好以后打上勾同意协议,开始下载安装


![](/Image/专业技能/TensorRT/TX2_step2.jpg)

**注意：这里选择安装的包中，包含安装Ubuntu系统，和cuda、cudnn、TensorRT等两部分，当TX2设备中已经包含Ubuntu系统后，可以不用选择安装jetson os,只需要选择安装其深度学习运行环境。**

![](/Image/专业技能/TensorRT/TX2_step2.jpg)


* step3：下载相关的软件包
* * 
* * 



