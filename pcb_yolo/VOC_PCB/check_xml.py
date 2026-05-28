#coding=utf-8
import os
import shutil
import random
from xml.etree.ElementTree import ElementTree,Element
import cv2

def read_xml(in_path):
  '''''读取并解析xml文件
    in_path: xml路径
    return: ElementTree'''
  tree = ElementTree()
  tree.parse(in_path)
  return tree

def check():
    url = "./Annotations" #修改成annotation的目�?
    for item in os.listdir(url):
        tree = read_xml(url + "/" + item)

        root = tree.getroot()
        object = root.findall("object")
        size = root.find("size")
        width =int(size.find("width").text)
        height = int(size.find("height").text)
        if object == None:
            print(item)
            continue
        for it in object:
            bndbox = it.find("bndbox")
            if bndbox == None:
                print(item)
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)
            if  xmin < 1 or xmin >= xmax or ymin >599 or ymin >= ymax:
            #if xmin <= 1 | xmin >= xmax:
                print("exceed bounding box---"+item)
            #if  xmin <= 1 or xmax <=1 or ymin <=1 or ymax <=1 :
            #if xmin <= 1 | xmin >= xmax:
                #print("exceed bounding box---"+item)
            if (xmin <=1 or xmax>=599 or ymax>=599 or ymin<=1)and(xmax-xmin<=5 or ymax-ymin<=5):
                print("bounding box too small---"+item)
            #if xmax >= width-2 or ymax>= height-2:
                #print("xmax or ymax exceed bounding box---"+item)

if __name__ =='__main__':
    check()


