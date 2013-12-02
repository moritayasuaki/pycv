#!/usr/bin/env python
# coding:utf-8
import sys
import socket
from bottle import route, run, template, request, static_file, redirect
from skimage import io
import os
import subprocess
import glob
import urllib2
import shutil
import json
import listfile

mountdir = "./data/"

def parentlink(path):
  parent = os.path.dirname(path)
  return '<a href="/dev_lcp/' + parent + '">' + '..' + '</a></br>\n'

def listing(path):
  fnames = os.listdir(mountdir + path)
  string = ''
  for fname in fnames :
    p = path + fname
    if os.path.isdir(mountdir + path + fname) :
      string += '<a href="/dev_lcp/' + path + fname  + '">' + fname + '</a></br>\n' 
    elif os.path.isfile(mountdir + path + fname) :
      base,ext = os.path.splitext(fname)
      if ext == '.jpg' or ext == '.html' or ext == '.png' or ext == '.json' :
        string += '<a href="/statics/' + path + fname + '">' + fname + '</a></br>\n'
      else :
        string += fname + '</br>\n'
  imgs = ''
  for fname in fnames :
    root, ext = os.path.splitext(fname) 
    if (ext == '.jpg' or ext == '.png'):
      imgs += '<img src="' + '/statics/' + path + fname + '" width="100" height="100">'
  return string + imgs

@route('/dev_lcp/')
def index(path=''):
  redirect('/dev_lcp')

@route('/dev_lcp')
def index(path=''):
  path = urllib2.unquote(path)
  html = '<h2> files in ' + path + '</h2>\n'
  html += listing(path)
  form =  '<form action="/exec/' + path + '" method="POST"><input value="Execute this directory" type="submit" /></form>'
  return html + form

@route('/statics/<filename:re:.*\.html>')
def html_show(filename):
  return static_file(filename, root=mountdir)

@route('/statics/<filename:re:.*\.json>')
def json_show(filename):
  return static_file(filename, root=mountdir)

@route('/statics/<filename:re:.*\.jpg>')
def send_image(filename):
  path = urllib2.unquote(filename)
  dirname = '/' + os.path.dirname(filename)
  basename = os.path.basename(filename)
  return static_file(filename, root=mountdir, mimetype='image/jpg')

@route('/statics/<filename:re:.*\.png>')
def send_image(filename):
  path = urllib2.unquote(filename)
  basename = os.path.basename(filename)
  return static_file(filename, root=mountdir, mimetype='image/png')

@route('/dev_lcp/<path:path>')
def index(path=''):
  path = urllib2.unquote(path)
  html = '<h2> files in ' + path + '</h2>\n'
  html += '<a href="/search/' + path  + '">' + 'search jpg under this directory' + '</a></br>\n' 
  html += parentlink(path) + listing(path + '/')
  form =  '<form action="/exec/' + path + '" method="POST"><input value="Execute this directory" type="submit" /></form>'
  return html + form

@route('/search/<path:path>')
def do_search(path):
  path = urllib2.unquote(path)
  paths = listfile.path_include_ext(mountdir + path,'.jpg')
  html = '<h2>jpg directory under ' + path + '</h2>\n'
  html += '<a href = "/dev_lcp/' + path + '">' + '.' + '</a></br>\n'
  for p in paths:
    pd = os.path.relpath(p,mountdir)
    pdd = os.path.relpath(p,mountdir + path)
    html += '<a href="/dev_lcp/' + pd  + '">' + pdd + '</a></br>\n' 
  return html

@route('/exec/<path:path>',method='POST')
def do_exec(path):
  upath = urllib2.unquote(path)
  print (mountdir + upath)
  subprocess.Popen(["./mark.py", mountdir + upath])
  shutil.copy("./result.html", mountdir + upath + '/result.html')
  datalist = [];
  redirect("/statics/" + path + "/result.html")

if len(sys.argv) == 2 :
  host = sys.argv[1]
else :
  host = socket.gethostbyname(socket.gethostname())
run(host=host, port=8080)

