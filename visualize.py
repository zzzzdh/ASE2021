import os
import csv
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/home/cheer/Project/VideoCaptioning/data'

def make_dir(video_name):
  output_path = os.path.join(data_dir, 'Visualize', video_name)
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  return output_path

def visualize_diff():
  diff_list = os.listdir(os.path.join(data_dir, 'Diff'))
  for diff_file in diff_list:
    with open(os.path.join(data_dir, 'Diff', diff_file), 'r') as dfile:
      lines = dfile.readlines()
  output_path = make_dir(diff_file.split('.')[0])
  for line in lines:
    image_name, x1, y1, x2, y2, cx, cy = line.split()
    image_name = os.path.join(data_dir, 'Images', diff_file.split('.')[0], image_name + '.png')
    image = cv2.imread(image_name)
    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, 2)
    image = cv2.circle(image, (int(cx), int(cy)), 5, (0, 255, 0), 2)
    cv2.imwrite(image_name.replace('Images', 'Visualize'), image)

def plot_year():
  image_path = os.listdir(os.path.join(data_dir, 'Images'))
  diff_path = [x.split('.')[0] for x in os.listdir(os.path.join(data_dir, 'Diff'))]
  csv_file = os.path.join(data_dir, '1732_items.csv')
  lines = []
  with open (csv_file, 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
      lines.append(row[4])
  years = []
  for line in lines[1:]:
    line = line.split('\n')[1]
    if re.search(r'[0-9]+ years ago', line):
      years.append(int(line.split()[0]))
    else:
      years.append(1)
  hist, edge = np.histogram(years, bins = range(1, 21, 1))
  plt.figure(figsize = (20, 15))
  axes1 = plt.subplot(111)
  width = 0.8
  len_range = np.arange(1, 20, 1)
  axes1.bar(np.arange(1, 20, 1), hist[::-1], width)
  plt.xticks(len_range, ('2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'), fontsize = 15)
  plt.yticks(fontsize = 15)

if __name__ == '__main__':
  plot_year()
