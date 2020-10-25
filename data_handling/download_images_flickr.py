#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:46:03 2020

@author: franciscapessanha
"""

import flickrapi
import urllib
from PIL import Image
import os
import requests
import os
import sys
import time

# From: https://gist.github.com/yunjey/14e3a069ad2aa3adf72dee93a53117d6

DATASET = os.path.join(os.getcwd(), 'dataset')

FLICKR = os.path.join(DATASET, 'flickr')

if os.path.exists(FLICKR) is not True:
	os.mkdir(FLICKR)


# Flickr api access key
flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)

keywords = ['stable', 'barn', 'field', 'farm']

for keyword in keywords:
	photos = flickr.walk(text=keyword,
	                     tag_mode='all',
	                     tags=keyword,
	                     extras='url_c',
	                     per_page=100,           # may be you can try different numbers..
	                     sort='relevance')

	urls = []
	for i, photo in enumerate(photos):
		print (i)

		url = photo.get('url_c')
		urls.append(url)

		# get 50 urls
		if i > 100:
			break

		# Download image from the url and save it to '00001.jpg'
		if url != None:
			path = os.path.join(FLICKR, '%s_%s.png' %(keyword, i))
			response=requests.get(url,stream=True)

		with open(path,'wb') as outfile:
			outfile.write(response.content)

		#urllib.request.urlretrieve(urls[1], path)

