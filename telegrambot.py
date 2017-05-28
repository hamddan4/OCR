# -*- coding: utf-8 -*-

import telepot
from telepot.loop import MessageLoop
#from telepot.namedtuple import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
import cv2
from main import process_img
import time
from skimage import io

from os import path, mkdir
import urllib

print('hello')



def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(content_type, chat_type, chat_id,msg['from']['first_name']+msg['from']['last_name'])



    if content_type == 'photo':
        message = " Let's see... "
        bot.sendMessage(chat_id, message)
        imdir = bot.getFile(msg["photo"][-1]["file_id"])
        print imdir
        num_proc = update_num_proc()
        url = "https://api.telegram.org/file/bot"+ TOKEN +"/"+ imdir["file_path"]
        print url
        urllib.urlretrieve(url, ImageFolder+"/foto_"+str(num_proc)+".jpg")

        im = io.imread(ImageFolder+"/foto_"+str(num_proc)+".jpg")
        
        text = process_img(im)
        
        textdir = TextFolder+"/test_"+str(num_proc)+".txt"
        
        if not path.isfile(textdir):
            textfile = open(textdir, 'w+')
            textfile.write(text)
            textfile.close()
        
        bot.sendMessage(chat_id,text)
        
    elif content_type == 'text':
        if  msg['text'] == "/start":
            
            message = 'Hello ' + msg['from']['first_name'] + ' ' + msg['from']['last_name'] + '\n'
            message += 'I\'m OCR bot, a bot that will try to figure out'
            message += 'letters in the images that you will send to me.\n'
            message += 'Send me an image and let me read it for you!'
            
            bot.sendMessage(chat_id, message)  
            
        elif  msg['text'] == "Hola Carambola":
            bot.sendMessage(chat_id, "hola pepsicola") 
        else:
            bot.sendMessage(chat_id, "Sorry, I didn't understand you") 
            
TOKEN = '340171475:AAFUDW_HiK1zaP-55plvA0zJNWaIVtQoYF8'
bot = telepot.Bot(TOKEN)
print(bot.getMe())



#==============================================================================
# im_num = 0
# filedir = 'telephoto/metadata.txt'
# with open(filedir, 'r') as myfile:
#     data=myfile.read()
#     myfile.close()
#     print data
#     im_num = int(data)
#     with open(filedir, 'w') as myfile:
#         myfile.write(im_num)
#         myfile.close()
# foto_name = "telegram-photo"+str(im_num)+".jpg"
#==============================================================================
#Data options
ImageFolder = 'Telegram-Images'
TextFolder = 'Telegram-Text'


def update_num_proc():
    #Creating root folder
    if not path.exists(TextFolder):
        mkdir(TextFolder)
    
    #Creating metadata
    if not path.isfile(TextFolder+"/metadata.txt"):
        metadata = open(TextFolder+"/metadata.txt", 'w+')
        metadata.write(str(1))
        metadata.close()
    
    metadata = open(TextFolder+"/metadata.txt", 'r+')
    
    num_proc = int([x for x in metadata][0])
    
    metadata.seek(0, 0);  
    metadata.write(str(num_proc+1))
    metadata.close()
    
    return num_proc
    
if not path.exists(ImageFolder):
    mkdir(ImageFolder)
if not path.exists(TextFolder):
    mkdir(TextFolder)
#===================================

MessageLoop(bot, handle).run_as_thread()
print('Listening ...')

while 1:
	time.sleep(10)
