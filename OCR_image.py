import tensorflow as tf
import sys
import os
#import glob
import subprocess


def percentage(part, whole):
  return 100 * float(part)/float(whole)


#input dir path
dir_path = sys.argv[1]
# imagePath = sys.argv[1]

#check if folder result and result.txt exist then delete, else create new
if os.path.exists("result_python_%s" %dir_path):
    os.system("rm -rf result_python_%s" %dir_path)
if os.path.isfile("result_python_%s.txt" %dir_path):
    os.system("rm -rf result_python_%s.txt" %dir_path)

os.system("mkdir result_python_%s" %dir_path)
print("\n create result_python_%s \n" %dir_path)

#list all pictures in dir input
#dir_filess = os.system("ls -v %s" %dir_path)
#dir_files = os.listdir(dir_path)
#dir_files = glob.glob("%s/*.jpg" %dir_path)
file_path = subprocess.Popen('ls -v %s/*.jpg' %dir_path, stdout=subprocess.PIPE, shell=True)
files_path = file_path.stdout.read()
files_paths = files_path.split()
#print files_paths


# if not tf.gfile.IsDirectory(imagePath):
#     tf.logging.fatal('imagePath directory does not exist %s', imagePath)


# # Get a list of all files in imagePath directory
# image_list = tf.gfile.ListDirectory(imagePath)
# print image_list

#Total pictures
Num=len(files_paths)
print("\n Total pictures in %s: %s \n"  %(dir_path,Num))


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
	           in tf.gfile.GFile("retrained_labels.txt")]


# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#line number of the document
line_num=0
line_temp=0
pic_num=0
#run
with tf.Session() as sess:

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    for image_path in files_paths:
    #for image_path in image_list:

        #print percent
        #percent=float(pic_num)/float(Num)
        print("Complete: %s%s" % (percentage(pic_num,Num),"%"))

        #input the previous line
        line_temp = line_num

                #get line number from picture's name
        line_num=image_path[len("%s/" %dir_path):]
        line_num=line_num[:line_num.find("_")]

        #if found a new line in document
        if line_num > line_temp :
            check_error=os.system("printf '\n' >> result_python_%s.txt" %dir_path)


        # Read in the image_data
        #image_data = tf.gfile.FastGFile("%s/%s" %(imagePath,image_path), 'rb').read()
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        # Feed the image_data as input to the graph and get first prediction
        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        node_id = top_k[0]
        human_string = label_lines[node_id]
        #score = predictions[0][node_id]
        #print('%s' %human_string)
        check_error=os.system("printf %s >> result_python_%s.txt" %(human_string,dir_path))

        #copy picture to output folder with the name of character
        pic_name="%s____%s" %(human_string,image_path[len("%s/" %dir_path):])
        check_error=os.system("cp %s result_python_%s/%s" %(image_path,dir_path,pic_name))

        #increase pic_num
        pic_num = pic_num + 1

        #    print('%s (score = %.5f)' % (human_string, score))
        #    for node_id in top_k:
        #        human_string = label_lines[node_id]
        #        score = predictions[0][node_id]
        #        print('%s (score = %.5f)' % (human_string, score))
