# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         videoChar
# Description:  Video To char
# Author:       逍遥
# Date:         2020/4/17
# -------------------------------------------------------------------------------
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageGrab, ImageFont


# import pygame


def readClipboard():
    """
    获取剪贴板图片
    :return:  Image.Image
    """
    img = ImageGrab.grabclipboard()
    if isinstance(img, Image.Image):
        # img.save('./img/test.png')
        return img


def imgToChar(imgPath, outPath):
    # 图片转字符
    if isinstance(imgPath, Image.Image):
        img = imgPath
    else:
        img = Image.open(imgPath)

    chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    uint = 256.0 / len(chars)

    w, h = img.size
    # 获取像素宽高
    if w > 150:
        n = w // 100
    else:
        n = 1
    w = w // n
    h = h // (n * 2)
    img = img.resize((w, h))
    text = ""

    for j in range(h):
        for i in range(w):
            r, g, b = img.getpixel((i, j))  # 获取像素点
            gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)  # 计算换成灰度
            char = chars[int(gray / uint)]
            text += char
        text += "\n"

    with open(outPath, 'w') as f:
        f.write(text)
    return text


def ArrayImgToColorCharImg(imgPath):
    # 数组的图片转换为彩色字符图
    """
    :param imgPath:  图片地址，或者cv2.img （数组）
    :param imgOutPath: 输出路径
    :return: cv2.img(array数组)
    """
    if isinstance(imgPath, np.ndarray):
        img = imgPath
    else:
        img = cv2.imread(imgPath)
    chars = "$@B%8&WM#oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1lI{}[]?i!<>+*~-;:,\"^`'. "
    chars = CharToImgArray(chars)
    uint = 256.0 / len(chars)

    Oh, Ow, p = img.shape  # 获取图片的高宽
    n1, n2, n3 = chars[0].shape  # 获取字符的高宽，用来重置图片高宽
    h = Oh / n1
    w = Ow / n2

    scale = 3  # 原图和字符画的比例  原图和字符画最终结果的比例
    h = h * scale
    w = w * scale

    # # 计算整个图片平均颜色  # 和循环外的颜色替换相关联
    # b, g, r = img.reshape(-1, 3).transpose()
    # f = lambda x: sum(x) / len(x)
    # bgr = (f(b), f(g), f(r))

    w, h = int(w), int(h)
    img = cv2.resize(img, (w, h))  # resize img图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 使用opencv转化成灰度图
    # gray = cv2.resize(gray, (w, h))  # resize灰度图
    # print(len(gray),len(gray[0]))
    # print(h,w)

    imgArrayAll = []  # 所有行的数组img
    for i in range(h):
        imgArray = []  # 每一行的数组
        for j in range(w):  # 字符串拼接
            char = chars[int(gray[i][j] / uint)]  # 确定使用那个字符
            bgr = img[i][j]  # 查看该像素颜色  提取原来该像素点的颜色
            ab, ag, ar = char.reshape(-1, 3).transpose()  # 将选中的字符 转换为原图像素的颜色
            color = (255, 255, 255)  # 字符中要替换的颜色 b,g,r
            # print(bgr)
            ab[ab != color[0]] = bgr[0]  # 若 不等于color指定的颜色则是替换 为原图像素点的颜色
            ag[ag != color[1]] = bgr[1]  # 若 不等于color指定的颜色则是替换
            ar[ar != color[2]] = bgr[2]  # 若 不等于color指定的颜色则是替换
            char = np.array([ab, ag, ar]).transpose().reshape((n1, n2, n3))
            imgArray.append(char)
        imgArrayAll.append(np.concatenate(imgArray, axis=1))
    imgR = np.concatenate(imgArrayAll, axis=0)

    # # 颜色替换  # 循环前面的 计算平均颜色关联
    # tShape = imgR.shape
    # ab, ag, ar = imgR.reshape(-1, 3).transpose()
    # ab[ab == 255] = bgr[0]
    # ag[ag == 255] = bgr[1]
    # ar[ar == 255] = bgr[2]
    # imgR = np.array([ab, ag, ar]).transpose().reshape(tShape)
    # # end

    imgR = cv2.resize(imgR, (Ow, Oh))
    # cv2.imwrite("./testData/test/test1.png", imgR)
    return imgR


def ArrayImgToCharImg(imgPath):
    # 数组的图片转换为字符图
    """
    :param imgPath:  图片地址，或者cv2.img （数组）
    :param imgOutPath: 输出路径
    :return: cv2.img(array数组)
    """
    if isinstance(imgPath, np.ndarray):
        img = imgPath
    else:
        img = cv2.imread(imgPath)
    chars = "$@B%8&WM#oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1lI{}[]?i!<>+*~-;:,\"^`'. "
    chars = CharToImgArray(chars)
    uint = 256.0 / len(chars)

    Oh, Ow, p = img.shape  # 获取图片的高宽
    n1, n2, n3 = chars[0].shape  # 获取字符的高宽，用来重置图片高宽
    h = Oh / n1
    w = Ow / n2

    scale = 3  # 原图和字符画的比例  原图和字符画最终结果的比例
    h = h * scale
    w = w * scale

    w, h = int(w), int(h)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 使用opencv转化成灰度图
    gray = cv2.resize(gray, (w, h))  # resize灰度图

    imgArrayAll = []  # 所有行的数组img
    for i in range(len(gray)):
        imgArray = []  # 每一行的数组
        for pixel in gray[i]:  # 字符串拼接
            imgArray.append(chars[int(pixel / uint)])
        imgArrayAll.append(np.concatenate(imgArray, axis=1))
    imgR = np.concatenate(imgArrayAll, axis=0)
    imgR = cv2.resize(imgR, (Ow, Oh))
    # cv2.imwrite("./testData/test/test1.png", imgR)
    return imgR


def CharToImgArray(char):
    """
    每张图片元素,可以通过调节这个生成的基础图片来修改最终生成的字符画的样子.
    出入字符串，就会生成每个字符串中每个字符的图片
    :param char: 输入字符串类似于：   "helloWord"
    :return:  图片的像素矩阵  cv2使用
    """
    reC = []
    plantType = "cv2"  # 确定使用cv2还是PIL
    if plantType == "cv2":
        w, h = 30, 35  # 生成每张图片的像素的 宽长
        r, g, b = 0, 0, 0  # 生成图片的颜色
        for c in char:
            # # ######## 创建空白图片 用来给cv2使用
            img = np.zeros((h, w, 3), np.uint8)  # 每一个字符的像素为 35*30  高35 宽30 使用 b,g,r
            img.fill(255)  # 使用白色填充图片区域,默认为黑色
            # r, g, b = random.randint(0,255),random.randint(0,255), random.randint(0,255)  # 生成图片的颜色(随机)
            img = cv2.putText(img, c, (0, 26), cv2.FONT_HERSHEY_SIMPLEX, 1, (b, g, r), thickness=4)  # rgb 顺序为 b g r
            # break
            # cv2.imwrite("./testData/test/%s.png" % c, img)
            reC.append(img)
        #  通过PIL绘制，然后转为 cv2可以识别的
    elif plantType == "PIL":
        w, h = 30, 30  # 生成每张图片的像素的 宽长
        r, g, b = 0, 0, 0  # 生成图片的颜色
        for c in char:
            img = Image.new("RGB", (w, h), (255, 255, 255))  # 创建空白图片 用来给Image使用
            dr = ImageDraw.Draw(img)
            font = ImageFont.truetype(os.path.join("c://window/fonts", "simsun.ttc"), 20)  # 设置字体及大小
            dr.text((0, 0), c, fill=(r, g, b), font=font)  # font=font,
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # cv2.imwrite("./testData/test/%s.png" % c, img)
            reC.append(img)
    return reC


def imgFilesToVideo(imgPath, size):
    # 图片文件列表 转视频

    # path = r'C:\Users\Administrator\Desktop\1\huaixiao\\'#文件路径
    filelist = os.listdir(imgPath)  # 获取该目录下的所有文件名
    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 12
    # size = (591,705) #图片的分辨率片
    file_path = r"./testData/video/" + str(int(time.time())) + ".mp4"  # 导出路径
    # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    fourcc = cv2.CV_FOURCC(*"mp4v")  # opencv版本是2
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') #opencv版本是3

    video = cv2.VideoWriter(file_path, fourcc, fps, size)

    for item in filelist:
        if item.endswith('.png'):  # 判断图片后缀是否是.png
            item = imgPath + '/' + item
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)  # 把图片写进视频
    video.release()  # 释放


def imgToVideo(imgList, outPath, fps=12):
    # Image 转视频
    """
    :param imgList:  图片列表 数组类型（cv2）
    :param outPath:  视频输出路径及名称
    :param fps:      帧率
    :return:  空
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    img = next(imgList)
    h, w, p = img.shape  # 获取图片的高宽
    print("输出视频宽：%s，高：%s，fps：%s" % (w, h, fps))
    widthHeight = (w, h)
    video = cv2.VideoWriter(outPath, fourcc, fps, widthHeight)  # 定义保存视频目录名称及压缩格式，fps=10,像素为1280*720
    video.write(img)
    # n=0
    for img in imgList:
        # cv2.imwrite("./testData/test/t%s.png"%n,img)
        # n += 1
        video.write(img)
    video.release()


def videoToFrame(videoPath, reType="cv2"):
    # 读取视频文件
    """
    :param videoPath: 视频地址
    :return: Image.Image
    """
    vc = cv2.VideoCapture(videoPath)
    # 通过摄像头的方式
    # vc=cv2.VideoCapture(1)
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
    width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    framesNum = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    yield {"widthHeight": (int(width), int(height)), "fps": fps, "framesNum": framesNum}
    if reType == "cv2":
        while rval:
            yield frame
            rval, frame = vc.read()
    elif reType == "PIL":
        while rval:
            # cv2.imshow("OpenCV", frame)
            # cv2.imwrite(pathName, frame)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            yield img
            rval, frame = vc.read()
    else:
        pass
    vc.release()


def videoToImgFile(videoPath, outPath):
    imgs = videoToFrame(videoPath, "cv2")
    n = 0
    print(next(imgs))
    for img in imgs:
        path = os.path.join(outPath, "%s.png" % n)
        cv2.imwrite(outPath + "%s.png" % n, img)
        n += 1
        if n % 100 == 0:
            print("已处理%s帧！" % n)


def videoToChar(videoPath):
    # 视频转字符
    """
    仅作为测试使用，视频生成字符串 放在list中
    :param videoPath:  视频地址
    :return: 无
    """
    show_heigth = 30
    show_width = 80

    ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
    # 生成一个ascii字符列表
    char_len = len(ascii_char)

    vc = cv2.VideoCapture(videoPath)  # 加载一个视频

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
    frame_count = 0
    outputList = []  # 初始化输出列表
    while rval:  # 循环读取视频帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 使用opencv转化成灰度图
        gray = cv2.resize(gray, (show_width, show_heigth))  # resize灰度图
        text = ""
        # # ######## 创建空白图片
        # # 使用Numpy创建一张A4(2105×1487)纸
        # img = np.zeros((1080, 1500, 3), np.uint8)
        # # 使用白色填充图片区域,默认为黑色
        # img.fill(255)
        # # ######## 创建空白图片

        for pixel_line in gray:
            for pixel in pixel_line:  # 字符串拼接
                text += ascii_char[int(pixel / 256 * char_len)]
            text += "\n"
        outputList.append(text)

        # y0, dy = 0, 25
        # for i, txt in enumerate(text.split('\n')):
        #     y = y0 + i * dy
        #     # print(txt)
        #     img = cv2.putText(img, txt, (0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        #
        #
        # cv2.imwrite("./testData/test/"+str(frame_count)+".png",img)
        # # input("ok")

        frame_count = frame_count + 1
        if frame_count % 100 == 0:
            print("已处理" + str(frame_count) + "帧")
        rval, frame = vc.read()
    print("处理完毕")

    for frame in outputList:
        os.system("cls")  # 清屏
        print(frame)
        print()
        print()


def videoToCharVideo(inPath, outPath):
    """
    视频转换为字符视频
    :param inPath:  视频地址
    :param outPath: 输出视频地址
    :return: 无
    """
    print("start")

    def t(inPath):
        frame = videoToFrame(inPath, reType="array")
        n = 0
        info = next(frame)  # 获取初始化信息，视频信息
        yield info  # return初始化信息，视频信息
        for img in frame:
            # a = np.array(imgToCharImg(img))
            a = cv2.cvtColor(np.asarray(ArrayImgToCharImg(img)), cv2.COLOR_RGB2BGR)
            # a = img
            n = n + 1
            if n % 100 == 0:
                print("已处理%s帧！" % n)
            yield a

    imgArray = t(inPath)
    info = next(imgArray)
    print("原始视频宽：%s，高：%s，fps：%s，帧数：%s" % (
        info["widthHeight"][0], info["widthHeight"][1], info["fps"], info["framesNum"],))
    imgToVideo(imgArray, outPath, fps=info["fps"])


def videoToColorCharVideo(inPath, outPath):
    """
    视频转换为彩色字符视频 每一帧颜色为平均页面总体的颜色
    :param inPath:  视频地址
    :param outPath: 输出视频地址
    :return: 无
    """
    print("start")

    def t(inPath):
        frame = videoToFrame(inPath, reType="cv2")
        n = 0
        info = next(frame)  # 获取初始化信息，视频信息
        yield info  # return初始化信息，视频信息
        for img in frame:
            # a = np.array(imgToCharImg(img))
            # a = cv2.cvtColor(np.asarray(ArrayImgToColorCharImg(img)), cv2.COLOR_RGB2BGR)  # 将Image.Image 图转换为cv2 的默认格式图
            a = ArrayImgToColorCharImg(img)
            n = n + 1
            if n % 100 == 0:
                print("已处理%s帧！" % n)
            yield a

    imgArray = t(inPath)
    info = next(imgArray)
    print("原始视频宽：%s，高：%s，fps：%s，帧数：%s" % (
        info["widthHeight"][0], info["widthHeight"][1], info["fps"], info["framesNum"],))
    imgToVideo(imgArray, outPath, fps=info["fps"])


if __name__ == '__main__':
    # img = readClipboard()
    # img.save('./testData/1.png')
    # text = imgToChar('./testData/1.png', './testData/test.txt')
    # imgToCharImg('./testData/1.png')
    # imgToColorCharImg('./testData/1.png')
    # charToImg(text)
    # videoToChar('./testData/1.mp4')
    # videoToCharVideo('./testData/1.mp4', './testData/out.mp4')
    # videoToImgFile('./testData/4.mp4', './testData/test/')
    videoToColorCharVideo('./testData/4.mp4', './testData/out4.mp4')  # 视频转换为彩色字符视频
    # a = ArrayImgToColorCharImg('./testData/2.png')
    # cv2.imwrite('./testData/output.png', a)
