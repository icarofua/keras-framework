#!/usr/bin/env python
# coding: utf-8

import itertools
import numpy as np
import os
import json
from sys import argv
import xml.dom.minidom as minidom
import xml.etree.cElementTree as cet
import xml.etree.ElementTree as et
from config import *

tam = int(argv[1])
count = int(argv[2])
types = ['', 
'sedan',
'suv',
'van',
'hatchback',
'mpv',
'pickup',
'bus',
'truck',
'estate']

colors=[
'',
'yellow',
'orange',
'green',
'gray',
'red',
'blue',
'white',
'golden',
'brown',
'black'
]

import logging as log

log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=log.INFO)

def combis(source, r=2, max_distance=1):
    for i, item in enumerate(source):
        for j in range(i+1, min(i+max_distance+1, len(source))):
            l = [item] + source[j:(j+r-1)]
            if len(l) == r:
                yield l

def read_xml(xml_file, folder):
    log.info("start read_xml: %s" % (xml_file))
    xmlp = et.XMLParser(encoding="utf-8")
    tree = et.parse(xml_file,parser=xmlp)
    root = tree.getroot()
    plates = {}
    total = 0
    for idoc in root:
        if idoc.tag == 'Items':
            for vehicle in idoc:
                vehID = int(vehicle.attrib.get('vehicleID'))
                cameraID = vehicle.attrib.get('cameraID')
                type1 = types[int(vehicle.attrib.get('typeID'))]
                color = colors[int(vehicle.attrib.get('colorID'))]
                imageName = vehicle.attrib.get('imageName')
                if not vehID in plates:
                    plates[vehID] = []
                plates[vehID].append({'cameraID':cameraID,'type':type1, 'color':color,'path':os.path.join(folder,imageName)})
                total +=1
    log.info("finish read_xml: %s" % (xml_file))
    return plates, total

def run_pos(dict1, amount):
    log.info("start run_pos %d" % (amount))
    n_samples = 0
    samples_set = []
    for v1,v2 in itertools.product(dict1,dict1):
        if n_samples >= amount:
            break
        if (v1 == v2):
            log.debug("run_pos %d %s %s" % (n_samples, v1, v2))
            comb1 = []
            comb2 = []
            if tam>1:
                comb1 = list(combis(dict1[v1], tam))
                comb2 = list(combis(dict1[v2], tam))
            else:
                comb1 = [[veh]for veh in dict1[v1]]
                comb2 = [[veh]for veh in dict1[v2]]

            type1 = POS
            comb1 = np.random.permutation(comb1).tolist()
            comb2 = np.random.permutation(comb2).tolist()
            samples = itertools.product(comb1,comb2)

            for car in samples:
                list_car0 = car[0]
                list_car1 = car[1]
                sample1 = [[],[],type1,[],[]]

                for car0,car1 in zip(list_car0, list_car1):
                    if car0['path'] == car1['path']:
                        continue

                    sample1[0].append(car0['path'])
                    sample1[1].append(car1['path'])
                    sample1[3].append({'cameraID':car0['cameraID'], 'color':car0['color'], 'type':car0['type']})
                    sample1[4].append({'cameraID':car1['cameraID'],'color':car1['color'], 'type':car1['type']})

                    if len(sample1[0])==tam:
                        log.debug("run_pos: add set %s" % (str(sample1)))
                        samples_set.append (sample1)
                        n_samples +=1

    log.info("finish run_pos %d" % (n_samples))
    return samples_set, n_samples


def run_neg(dict1, amount):
    log.info("start run_neg %d" % (amount))
    n_samples = 0
    samples_set = []
    labels_set = []
    options = np.random.permutation(list(itertools.product(list(dict1),list(dict1)))).tolist()
    opts = list(range(len(options)))

    while (n_samples < amount and opts != []):
        opt = np.random.choice(opts)
        v1,v2 = options[opt]
        log.debug("run_neg option %d %s %s" % (opt, v1, v2))

        if v1 ==v2 or (v1,v2) in labels_set or (v2,v1) in labels_set:
            opts.remove(opt)
            continue
    
        opts.remove(opt)
        labels_set.append((v1,v2))

        comb1 = []
        comb2 = []
        type1 = NEG
        if tam>1:
            comb1 = list(combis(dict1[v1], tam))
            comb2 = list(combis(dict1[v2], tam))
        else:
            comb1 = [[veh]for veh in dict1[v1]]
            comb2 = [[veh]for veh in dict1[v2]]

        #comb1 = [comb1[int(len(comb1)/2)]]
        #comb2 = [comb2[int(len(comb2)/2)]]
        min_tam = min(len(comb1), len(comb2))
        samples = zip(comb1[:min_tam], comb2[:min_tam])
        #samples = itertools.product(comb1,comb2)
        
        for car in samples:
            list_car0 = car[0]
            list_car1 = car[1]
            sample1 = [[],[],type1,[],[]]

            for car0,car1 in zip(list_car0, list_car1):
                sample1[0].append(car0['path'])
                sample1[1].append(car1['path'])
                sample1[3].append({'cameraID':car0['cameraID'], 'color':car0['color'], 'type':car0['type']})
                sample1[4].append({'cameraID':car1['cameraID'],'color':car1['color'], 'type':car1['type']})

                if len(sample1[0])==tam:
                    log.debug("run_neg set %s" % (str(sample1)))
                    samples_set.append (sample1)
                    n_samples +=1

    log.info("finish run_neg %d" % (n_samples))

    return samples_set, n_samples

def checkID(set1, pos1, pos2):
    return set1[pos1][0][0].split("/")[-1].split("_")[0] == set1[pos2][0][0].split("/")[-1].split("_")[0]


def checkPart(samples, n_samples, part1):
    if checkID(samples, part1-1, part1):
        #check if the first vehicleID from validation is the same vehicleID from train
        count_back = 0
        count_forward = 0
        back = part1-1
        forward = part1+1

        while (back > 0 and checkID(samples, back, part1)):
            count_back +=1
            back -=1

        while (forward < n_samples and checkID(samples, part1, forward)):
            count_forward +=1
            forward +=1

        if (count_back < count_forward or forward == n_samples):
            part1 = back
        else:
            part1 = forward                
    return part1

xml1,total1 = read_xml('train_label.xml', 'image_train')
xml2,total2 = read_xml('test_label.xml', 'image_test')

dataset = {}
dict_aux = {'train':xml1, 'test':xml2}
for k,xml in dict_aux.items():
    samples_set_pos, n_samples_pos = run_pos(xml,count + int(0.2*count))
    count = min(count, n_samples_pos)
    samples_set_neg, n_samples_neg = run_neg(xml, n_samples_pos)  
    if k == 'train':
        part1Pos = int(0.8*n_samples_pos)
        part1Neg = int(0.8*n_samples_neg)
        part1Pos = checkPart(samples_set_pos, n_samples_pos, part1Pos)
        part1Neg = checkPart(samples_set_neg, n_samples_neg, part1Neg)
        dataset[k] = np.random.permutation(samples_set_pos[:part1Pos] + samples_set_neg[:part1Neg]).tolist()
        dataset['validation'] = samples_set_pos[part1Pos:] + samples_set_neg[part1Neg:]
    else:
        dataset[k] = samples_set_pos + samples_set_neg
    log.info("%s neg:%d pos:%d\n" % (k, n_samples_neg, n_samples_pos))
with open('dataset%d_%d.json' % (count,tam), 'w') as fp:
    json.dump(dataset, fp)
