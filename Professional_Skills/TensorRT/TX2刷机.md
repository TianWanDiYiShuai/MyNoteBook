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

  * 下载软件包
  * 主机设备与TX2设备连接

    * 连接网线，需要设备与主机在同一个网关下，能相互ping通
    * 连接鼠标与键盘
    * 连接电源线
    * USB线与主机连接
    * 设备进入Recovery模式，同时按下并保存RECOVERY键S3，与RESET键S1；然后松开RESET键S1，最后保持按住RECOVERY键S3，2秒后松开RECOVERY键S3。
    * 输入连接设备的账号：nvidia，密码：nvidia，ip：....

**安装分为，自动安装和手动安装，自动安装，需要输入设备的ip,账号密码，手动安装，需要ip,账号密码，并手动进入Recovery模式。**

* step4：安装成功，退出。

## 2、安装环境说明

![](/Image/专业技能/TensorRT/TX2版本信息.png)

* cuda ：10.0
* cudnn：7.5.0
* TensorRT：5.1.6
* opencv：3.3.1



