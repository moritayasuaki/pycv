#!/usr/bin/env python
# coding:utf-8
from bottle import route, run, template, request, static_file
from skimage import io
import os
import glob
import urllib2
import mark
import shutil
import json

mountdir = "./data/"

def parentlink(path):
  parent = os.path.dirname(os.path.dirname(path))
  return '<a href="/dev_lcp' + parent + '">' + '..' + '</a></br>\n'

def listing(path):
  fnames = os.listdir(mountdir + path)
  string = ''
  for fname in fnames :
    p = path + fname
    if os.path.isdir(mountdir + path + fname) :
      string = string + '<a href="/dev_lcp/' + path + fname  + '">' + fname + '</a></br>\n' 
    elif os.path.isfile(mountdir + path + fname) :
      string = string + fname + '</br>\n'
  imgs = ''
  for fname in fnames :
    root, ext = os.path.splitext(fname) 
    if (ext == '.jpg' or ext == '.png'):
      imgs += '<img src="' + '/images/' + path + fname + '" width="100" height="100">'
  return string + imgs

@route('/dev_lcp/')
def index(path=''):
  form =  '<form action="/exec/' + path + '" method="POST"><input value="Execute this directory" type="submit" /></form>'
  html = listing(path + '/')
  return form + html

@route('/dev_lcp')
def index(path=''):
  path = urllib2.unquote(path)
  form =  '<form action="/exec/' + path + '" method="POST"><input value="Execute this directory" type="submit" /></form>'
  html = listing(path + '/')
  return form + html

@route('/images/<filename:re:.*\.jpg>')
def send_image(filename):
  path = urllib2.unquote(filename)
  dirname = '/' + os.path.dirname(filename)
  basename = os.path.basename(filename)
  return static_file(filename, root=mountdir, mimetype='image/jpg')

@route('/images/<filename:re:.*\.png>')
def send_image(filename):
  path = urllib2.unquote(filename)
  basename = os.path.basename(filename)
  return static_file(filename, root=mountdir, mimetype='image/png')

@route('/dev_lcp/<path:path>')
def index(path=''):
  path = urllib2.unquote(path)
  html = parentlink(path) + listing(path + '/')
  form =  '<form action="/exec/' + path + '" method="POST"><input value="Execute this directory" type="submit" /></form>'
  return form + html


@route('/exec/<path:path>',method='POST')
def do_exec(path):
  path = urllib2.unquote(path)
  print path
  jpgs = glob.glob(mountdir + path + '/*.jpg')
  print "do_exec"
  datalist = [];
  for jpg in jpgs :
    fname = os.path.basename(jpg)
    image = io.imread(jpg,as_grey=True)
    cimage,rel,r,center,p0,p1 = mark.run(image)
    data = { 'fname':fname,
             'width':image.shape[1],
             'height':image.shape[0],
             'rel':rel,
             'r':r,
             'cx':center[1],
             'cy':center[0],
             'px0':p0[1],
             'py0':p0[0],
             'px1':p1[1],
             'py1':p1[0] }
    datalist.append(data)
  js = json.dumps(datalist)
  fd = open(mountdir + path + '/result.json','w')
  fd.write(js)
  fd.close()
  shutil.copy("./result.html", mountdir + path + '/result.html')
  return js

run(host='localhost', port=8080)

