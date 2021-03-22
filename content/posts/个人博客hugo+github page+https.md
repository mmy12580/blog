---

title: "网站搭建:hugo+github"
date: 2019-03-01T11:27:28-05:00
draft: false
tags: ['github']
categories: ['website']
---

# 初衷 

个人网站这个事情，想倒腾很久了。可惜一直被各种事情给影响，近来想着，还是得发一下狠。在2019年年初倒腾一个个人网站，原因很简单，高效的做做笔记，发表一些看法，希望能和更多人交流，学习以及成长。Stay foolish, stay hungary!
本文将介绍如何搭配Hugo + Github Pages + 个人域名的流程。因为我是用Mac搭建的，所以这里的准备工作和具体的流程都只包含了如何用Mac搭建（linux 大同小异)。这里对windows的童鞋先说声抱歉了(シ_ _)シ，因为我学代码开始没用过:sweat_smile:。对于写代码的要求，这里并不高，只需要你对terminal会用一些常用的代码就可以了，当然，其最基本的git的代码还是需要的 e.g git clone, add, commit, push这些 。而对于完全没写过代码的小白，有一些东西也只能麻烦你们自己google了，比如如何建立github。我这里会提供一些相对应的链接，以方便你在建立网站时的流程.



## 准备工作

正如标题所说，只需要安装hugo, github page, 以及https保障网站安全就好了.

### 依赖环境：

* brew
* git
* hugo

### 前期安装 

安装brew, 先打开`spotlight`输入`terminal`, 然后复制以下代码

```nolinenumbers
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
安装后，安装git

```nolinenumbers 
brew install git
```

安装我们需要的网站建立的框架

```nolinenumbers
brew install hugo
```
选择管理blog的位置,例如我的桌面，然后建立新项目e.g myblog, 并进入blog文件夹

```nolinenumbers
cd ~/Desktop
hugo new site myblog
cd myblog
```
尝试建立内容为”hello world"的post, 将其命名为myfirst_post.md

```nolinenumbers
hugo new posts/myfirst_post
echo "hello world" > content/posts/myfirst_post.md
```
启动hugo的静态服务:

```nolinenumbers
hugo sever -D
```
这个时候会显示一对代码例如:

```plain
Serving pages from memory Running in Fast Render Mode. For full rebuilds on change: hugo server --disableFastRender
Web Server is available at http://localhost:1313/ (bind address 127.0.0.1)
Press Ctrl+C to stop
```

打开浏览器，进入localhost:1313就能看到你的网站内容了;

## Host on Github

这个时候的简单博客只适合在本地使用，是一个可以写完博客，并且查看所得内容的呈现，但是想要给其他人看，需要做成一个网站。作为一名程序猿，github再适合不过了。
这里特指github page。建立github page, 可以说极其简单，直接参照[官网](https://pages.github.com/)的第一步，进入github, 创建新的repo, 为其命名xxx.github.io. xxx要对应你的github的账号名。

![create_repo](/images/create_repo.png)

接下来就只需要做两件事情。

1. 将刚刚生成的blog（整个文档）做成一个github repo。将其命名为 xxxblog
2. 在生成的xxblog里，将github page repo 例如 xxx.github.io, 生成在 xxxblog里

步骤一的方法可以直接参考[将已存在目录转换为git repo](http://leonshi.com/2016/02/01/add-existing-project-to-github/)。
完成后在目录内，运行

```
git submodule add -b master git@github.com:<USERNAME>/<USERNAME>.github.io.git public. 
```
**note**: 这里的`<USERNAME>`指的是你github账号的名字
这行代码的意义是将你的github.io，也就是github page作为运行你的博客的host，等会可以连接你发布的静态文档，以方便其他人和自己在不同的网络里登陆并且阅读
因为，发博客是持续性的工作，所以为了简单化发博客的特点，这里特地加了一个脚本(script)，以方便每次只需要将写好的markdown，commit到host（xxx.github.io)上。

在当前目录下, 建立一个可执行文件 deploy.sh。将以下内容复制到deploy.sh上。

```
#!/bin/bash
echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# 启动hugo.
hugo 

# 进入public 文件夹，这个实际上是xxx.github.io
cd public

# 加入新发布的markdown
git add .

# 标注此次更新的内容与时间 
msg="rebuilding site `date`"
if [ $# -eq 1 ]
  then msg="$1"
fi
git commit -m "$msg"

# 上传到xxx.github.io
git push origin master

# 返回上一级目录
cd ..
```

将deploy.sh，变成可执行文件
```
chmod +x deploy.sh
```

**谨记**：将public添加到blog里面的./gitignore，这样不会影响到repo的问题。如果没有gitignore, 可以直接创立如下.
```
echo public/ > .gitignore
```

大功告成，以后写新的博客以及发布只需要像一下一样
```
# 进入blog所在目录
cd blog

# 创建新博客例如 深度学习笔记
hugo new posts/深度学习笔记.md

# 运行deploy.sh 发布到自己的Host上
./deploy.sh
```

## 进阶设置

### 主题设置
刚刚的演示只是建立了一个小白板的过程，一个让人眼前一亮的UI，也是很需要的。可以去[hugo主题](https://themes.gohugo.io)，下载你喜欢的主题，并放入`theme/`目录下
。然后更改你的`config.toml`. 运行`hugo server -D`，在本地查看效果以方便调整

```
#将hugo那行改成 你下载的主题 例如Serif
hugo -t Serif
```
如果你感兴趣我的主题，可以去下载[LeaveIt](https://themes.gohugo.io/leaveit/)

### 自定义域名
很多人感觉访问自己的博客 xxx.github.io不够酷，这里有两种方案，

1. 免费一年的域名，从[FreeNom](https://www.freenom.com/en/index.html?lang=en), 可获得 .TK / .ML / .GA / .CF / .GQ
2. 付费域名 e.g Godaddy, Domain.com,各种

因为一致做机器学习，所以看到.ml很合适，我就选择了freenom，申请了自己[博客](moyan.ml)的地址。申请很简单，直接按照官网步骤走，弄好之后，
我们来连接xxx.github.io以及自己的域名例如xxx.ml。
```
# 进入github.io即为public的文件夹下
cd blog/public

# 创立一个文件并将申请的域名写入
echo xxx.ml > CNAME

# 复制github.io对应的地址
ping xxx.github.ml
```

然后我们进入freenom网站，添加一下xxx.github.io和xxx.ml的关系。
先进入MyDomains -> Manage Domain -> Management Tools -> NameServers把DNSPod中刚刚生成出的两个记录例如 A:192.30.252.153 和 CNAME:xxx.ml。刚刚生成的CNAME自动会将xxx.github.io转为刚刚申请的xxx.ml。保存后过，运气好的话10分钟，否则可能是几个小时，点开xxx.ml即可使用。如果你登陆过发现，当时显示404无法访问之类的错误，删除cookie,重新登录，就可以进入你的网站了。

Notes: Freenom一年免费，然后你可以免费续，邮箱会在最后一个月收到。

### 添加https

我用的[cloudflaree](https://www.cloudflare.com/en-ca/), 完全免费，步骤非常简单。首先注册账户，然后添加你的网站xxx.ml, cloudfiare会自动搜索到你的网站，然后你就下一步。选择always https为True, 然后下一步，生成完毕后。会在最后一页提供给你新的name sever, 你再次进入freenom的use custom nameservers，将这些新的name sever复制粘贴上去。直到cloudflare给你发邮件你的xxx.ml is active. 其实添加https意义并不大，个人主页基本上其实也是分享一些东西，几乎也不会有任何攻击，可能更多是某些浏览器添加了一些防病毒以及网页保护的工具，会因为不是https无法访问。当然，写博客还是希望大家能看到的，所以费一点时，能给自己和大家带来不必要的麻烦。


