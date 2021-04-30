import csv
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
#
from django.core.files.storage import FileSystemStorage
from os import listdir
from os.path import isfile, join

    

def get_all_values(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from get_all_values(v)
    elif isinstance(d, list):
        for v in d:
            yield from get_all_values(v)
    else:
        yield d 
    return None

def dataR(data):
    datan=list(get_all_values(data))
    print(datan)
    datan=datan[1:-1]
    field=['type','name','fname','mname','age','gender','dob','address','pincode','eyeimg','skinimg','submit','csrfmiddlewaretoken']
    datan.append(1)
    print("______",datan,"________________")
    with open('./media/csvdata/Patient.csv','a') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(datan)
    return 0

def getid():
    onlyfiles = [f for f in listdir('./media/img') if isfile(join('./media/img', f))]
    # print(onlyfiles)
    l=len(onlyfiles)
    return l

def fdata(datan):
    fs=FileSystemStorage()
    print(datan)
    f=datan['eyeimg']
    id=str(getid())
    imgformat='img/'+id+'.png'
    path = default_storage.save(imgformat, ContentFile(f.read()))
    tmp_file = os.path.join(settings.MEDIA_ROOT, path)  
    print(f)
    data=list(get_all_values(datan))
    print(data)

    # filname,ext=str(f).split('.')
    # file=fs.save(str(f),f)
    # fileurl=fs.url(file)
    # size=fs.size(file)
    # print(fileurl,"************************",size)
    # datan=list(get_all_values(data))
    # datan=datan[1:-1]
    # print(datan)

    return id

def getdata():
    f=open('./media/csvdata/Patient.csv','rt')
    data=csv.reader(f)
    for row in data:
        print(row)
        last=row
    return last    
    



