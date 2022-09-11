import ghostscript
from PyPDF2 import PdfFileWriter, PdfFileReader
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from tqdm import tqdm
import shutil

import torch
from classifier import NET, NET_


# min: 5800
# max: 7800


# OPTION 1: Training DATA Extractor from pdf
class Question_Extractor():
    def __init__(self, folder_path, h_rm, v_rm, reset_save = False, min_threshold_area = 4000, max_threshold_area = 8000, crp_pct = 0.7,keep_raw_photo_for_check = True, remove_front = 1, remove_back = 1):
        self.folder_path = folder_path
        self.h_rm = h_rm
        self.v_rm = v_rm
        self.remove_front = remove_front
        self.remove_back = remove_back
        self.keep_raw_photo_for_check = keep_raw_photo_for_check
        self.train_dat_crop_pct = crp_pct
        self.min_threshold_area = min_threshold_area
        self.max_threshold_area = max_threshold_area
        self.save_values = {}
        self.reset_save = reset_save
        self.h_pct = 0.55

        self.name_counter = 0
        self.dat_num = 0
        self.bottom = 0
        self.qs_ID = 0


        self.qs = []
        self.current_mark = 0


        # Initialize the Recgniser and Differentiator Neural Net
        self.markDifferentiator = NET_()
        self.markRecogniser = NET()
        self.mark_label = torch.argmax(torch.Tensor([0.,1.]))
        
        with open("models/Recogniser_Neural_Net-v4.pickle", 'rb') as f:
            self.markRecogniser = pickle.load(f)

        with open("models/Differentiator_Neural_Net-v4.pickle", 'rb') as f:
            self.markDifferentiator = pickle.load(f)

        if os.path.exists("paramDat.pickle") and not self.reset_save:
            with open("paramDat.pickle", 'rb') as p:
                self.save_values = pickle.load(p)


    def pdf2jpeg(self, pdf_input_path, jpeg_output_path):
        args = ["pdf2jpeg", # actual value doesn't matter
            "-dNOPAUSE",
            "-sDEVICE=jpeg",
            "-r144",
            "-sOutputFile=" + jpeg_output_path,
            pdf_input_path]
        ghostscript.Ghostscript(*args)


    def convertPdf2Photos(self, input_pdf):
        input_pdf = PdfFileReader(open(input_pdf, 'rb'))
        for i in range(input_pdf.numPages):
            out = PdfFileWriter()
            out.addPage(input_pdf.getPage(i))
            with open(f"{self.folder_path}/Lab/doc-{i}.pdf", 'wb') as outputStream:
                out.write(outputStream)

            self.pdf2jpeg(f"{self.folder_path}/Lab/doc-{i}.pdf", f"{self.folder_path}/Lab/{i}.jpeg")
            os.remove(f"{self.folder_path}/Lab/doc-{i}.pdf")


    def crop_save(self):
        for image in glob.glob(f'{self.folder_path}/Lab/*.jpeg'):
            img = cv2.imread(image)
            cropped = img[self.v_rm:img.shape[0]-self.v_rm, self.h_rm:img.shape[1]-self.h_rm]
            cv2.imwrite(image,cropped)


    def is_mark(self, img):
        img = torch.Tensor(img).view(20, 105)/255.0
        with torch.no_grad():
            # print(img.view(-1, 1, 73, 20).shape)
            out = torch.argmax(self.markRecogniser(img.view(-1, 1, 20, 105)))

        if out == self.mark_label:
            return True
        else:
            return False


    def differentiator(self,img):
        img = torch.Tensor(img).view(20, 32)/255.0
        with torch.no_grad():
            # print(img.view(-1,1,32, 20))
            out = torch.argmax(self.markDifferentiator(img.view(-1,1,20, 32)))
            # print(torch.argmax(out))
        return out


    def reverse_sort_img_files(self, img_files):
        files = []
        path_origin = ''
        for i in img_files:
            split_list = i.split('/')
            files.append(int( split_list[-1].replace(".jpeg", '') ))
            
            if path_origin == '':
                for x in split_list[:-1]:
                    path_origin+=x+'/'
        

        files = sorted(files, reverse = True)
        # del split_list

        img_files = [path_origin+str(i)+".jpeg" for i in files]

        return img_files


    def extract_training_data(self, d_t, training_dat):
        # out = self.differentiator(d_t)+1
        # cv2.imwrite(f'{self.folder_path}/training_images/{out}_{self.name_counter}.jpeg', training_dat)#(f'{self.folder_path}/training_images/{self.dat_num}_detected.jpeg', training_dat)#
        # self.name_counter+=1
        cv2.imwrite(f'{self.folder_path}/training_images/{self.dat_num}_detected.jpeg', training_dat)
        self.dat_num+=1





    def extract_questions(self, dim_list, img):
        if not self.first:
            self.qs.append(img[ dim_list[1]+dim_list[3]: self.bottom, 0: img.shape[1] ])
        else:
            self.first = False

        self.bottom = dim_list[1]+dim_list[3]


    def save_extracted_qs(self,):
        # for qs in reversed(self.qs):
        #     plt.imshow(qs)
        #     plt.show()
        length = len(self.qs)
        for i in range(length-1, -1, -1):
            cv2.imwrite(f"Lab/extracted/{self.qs_ID}_{length-i}_{self.current_mark}.jpeg", self.qs[i])
        self.qs = []
        self.current_mark = 0
        self.qs_ID+=5
                    
    



    def extract_data(self, extract_t_d):
        img_folder = []
        for image in glob.glob(f"{self.folder_path}/Lab/*.jpeg"):
            img_folder.append(image)

        # if not extract_t_d:
        img_folder = self.reverse_sort_img_files(img_folder)



        # initialize the saved data from "ParamDat.pickle" file
        if "idxTraining" in self.save_values.keys():
            self.dat_num = self.save_values["idxTraining"]
        else:
            self.dat_num = 1

        if "ID" in self.save_values.keys():
            self.qs_ID = self.save_values["ID"]+5
        else:
            self.qs_ID = 1



        crops = []
        for i in tqdm(range(self.remove_back, len(img_folder)-self.remove_front)):
            img = cv2.imread(f"{img_folder[i]}")
            self.bottom = img.shape[0]
            self.first = True

            crp_h = int(img.shape[1]*self.h_pct)
            crops = [img[0:img.shape[0], 0:crp_h], img[0:img.shape[0], crp_h:img.shape[1]]]
            
            rgb = crops[1]
            # rgb = img



            # removing all the horizontal lines to improve accuracy
            img_gry = cv2.cvtColor(rgb.copy(), cv2.COLOR_BGR2GRAY)
            img_cny = cv2.Canny(img_gry, 200, 300)
            
            lns = cv2.ximgproc.createFastLineDetector().detect(img_gry)

            tolerance = 5
            img_cpy = rgb.copy()
            if lns is not None:
                for ln in lns:
                    x1 = int(ln[0][0])
                    y1 = int(ln[0][1])
                    x2 = int(ln[0][2])
                    y2 = int(ln[0][3])

                    if ((y2 == y1) or (y2-tolerance<=y1<=y2+tolerance) or (y1-tolerance <=y2<=y1+tolerance)) and abs(x1-x2)>20:
                        cv2.line(img_cpy, pt1=(x1, y1), pt2=(x2, y2),
                                 color=(255, 255, 255), thickness=3)
            
            # Extracting the likely "mark" lines
            small = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2GRAY)

            # NOTE: threshold the image
            _, bw = cv2.threshold(small, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # NOTE: get horizontal mask of large size since text are horizontal components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


            contours, hierarchy,=cv2.findContours(connected.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
 
            # NOTE: Countour detection with area restriction


            final_blocks = []
            for j in contours:
                x,y,w,h = cv2.boundingRect(j)
                if (w*h)>self.min_threshold_area and (w*h)<self.max_threshold_area and h<30:
                    final_blocks.append(j)

            # NOTE: Segment the text lines
            # NOTE: change this var to countours for non restricted marking
            blocks = final_blocks
            has_mark = False
            for idx in range(len(blocks)):
                x, y, w, h = cv2.boundingRect(blocks[idx])
                training_dat = img_cpy[y:y+h, x+int(w*self.train_dat_crop_pct):x+w]
               

                # # NOTE: r_t is the data that needs to be passed through the recogniser net
                # # NOTE: d_t is the data that need to be passed through the differentiator net
                if os.path.exists("Lab/temp.jpeg"):
                    os.remove("Lab/temp.jpeg")
                cv2.imwrite("Lab/temp.jpeg", training_dat)
                d_t = cv2.imread("Lab/temp.jpeg", cv2.IMREAD_GRAYSCALE)
                # r_t = cv2.resize(d_t[0:d_t.shape[0]-2, int(d_t.shape[1]*0.26):d_t.shape[1]],  (73, 20))
                r_t = cv2.resize(d_t, (105, 20))
                d_t = cv2.resize(d_t[0:d_t.shape[0], 0:int(d_t.shape[1]*0.28)], (32,20))

                # cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 0, 255), 2)
                # plt.imshow(r_t)
                # plt.show()

                if self.is_mark(r_t):
                    has_mark = True



                    cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 0, 255), 2)
                    # plt.imshow(d_t)
                    # plt.show()

                    if extract_t_d:
                        self.extract_training_data(d_t, training_dat)
                    else:
                        # NOTE: for now we'll extract training data while extracting qs only for olvl papers, to improve the model
                        # self.extract_training_data(d_t, training_dat)

                        if self.qs==[] and not self.current_mark:
                            self.current_mark = self.differentiator(d_t)+1

                        self.extract_questions([x,y,w,h], img)
                        if self.qs!=[]:
                            self.save_extracted_qs()
                            self.current_mark = self.differentiator(d_t)+1


            if not extract_t_d:
                if not has_mark:
                    self.qs.append(img)
                else:
                    self.qs.append(img[0 : self.bottom, 0: img.shape[1]])

            

            
            self.bottom = img.shape[0]


            marked_page = np.concatenate((crops[0], crops[1]), axis = 1)
            cv2.imwrite(f'{img_folder[i]}', marked_page)
        
            # Saving the crop ID and training img ID
            self.save_values["idxTraining"] = self.dat_num
            self.save_values["ID"] = self.qs_ID
            with open("paramDat.pickle", 'wb') as paramFile:
                pickle.dump(self.save_values, paramFile)
            paramFile.close()

        self.save_extracted_qs()
           

        if not self.keep_raw_photo_for_check:
            for im in img_folder:
                os.remove(f"{self.folder_path}/Lab/{im}")
                
        if os.path.exists("Lab/temp.jpeg"):
                    os.remove("Lab/temp.jpeg")



class rewiew_pages():
    def __init__(self, folder_path, h_rm, v_rm, keep_raw_photo_for_check = True):
        self.folder_path = folder_path
        self.h_rm = h_rm
        self.v_rm = v_rm
        self.keep_raw_photo_for_check = keep_raw_photo_for_check

    def pdf2jpeg(self, pdf_input_path, jpeg_output_path):
        args = ["pdf2jpeg", # actual value doesn't matter
            "-dNOPAUSE",
            "-sDEVICE=jpeg",
            "-r144",
            "-sOutputFile=" + jpeg_output_path,
            pdf_input_path]
        ghostscript.Ghostscript(*args)


    def convertPdf2Photos(self, input_pdf, n_page = 2):
        input_pdf = PdfFileReader(open(input_pdf, 'rb'))
        for i in range(n_page):
            out = PdfFileWriter()
            out.addPage(input_pdf.getPage(i))
            with open(f"{self.folder_path}/Lab/doc-{i}.pdf", 'wb') as outputStream:
                out.write(outputStream)

            self.pdf2jpeg(f"{self.folder_path}/Lab/doc-{i}.pdf", f"{self.folder_path}/Lab/{i}.jpeg")
            os.remove(f"{self.folder_path}/Lab/doc-{i}.pdf")
            # time.sleep(0.01)


    def crop_save(self, pause_time = 2.5):
        img = cv2.imread(f'{self.folder_path}/Lab/1.jpeg')
        cropped = img[self.v_rm:img.shape[0]-self.v_rm, self.h_rm:img.shape[1]-self.h_rm]
        plt.imshow(cropped)
        plt.show(block = False)
        plt.pause(pause_time)
        plt.close()
        os.remove(f'{self.folder_path}/Lab/1.jpeg')



# OPTION 2: AUGMENT EXTRACTED DATA
class Augmentor:
    def __init__(self, folder_path, number_of_files_required = 100, brightness_list = [310, 380], contrast_list = [175]):
        self.brightness_list = brightness_list
        self.contrast_list = contrast_list
        self.folder_path = folder_path
        self.number_of_files_required = number_of_files_required
        self.initial_files = []
        self.num_initial_files = len(self.initial_files)
        self.avg_shape = (0,0)
        self.ParamFiledata = {}
        if os.path.exists("paramDat.pickle"):
            with open("paramDat.pickle", 'rb') as p:
                self.ParamFiledata = pickle.load(p)
        p.close()

        self.get_files_in_directory()
        self.get_average_shape()

    def controller(self, img, brightness=255,contrast=127):
        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max = 255
            else:
                shadow = 0
                max = 255 + brightness
            al_pha = (max - shadow) / 255
            ga_mma = shadow

            # The function addWeighted calculates
            # the weighted sum of two arrays
            cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma)
        else:
            cal = img
        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)

            # The function addWeighted calculates
            # the weighted sum of two arrays
            cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)
        return cal



    def get_files_in_directory(self):
        self.initial_files = []
        for img in glob.glob(f"{self.folder_path}/*.jpeg"):
            self.initial_files.append(img)
        self.num_initial_files = len(self.initial_files)




    def save_average_shape(self,):
        if "Avg_shape_of_training_data" in self.ParamFiledata.keys():
            x = self.ParamFiledata["Avg_shape_of_training_data"]
            self.avg_shape = ( (self.avg_shape[0]+x[0])//2, (self.avg_shape[1]+x[1])//2 )
            self.ParamFiledata["Avg_shape_of_training_data"] = self.avg_shape
        else:
            self.ParamFiledata["Avg_shape_of_training_data"] = self.avg_shape

        # if os.path.exists("paramDat.pickle"):
        with open("paramDat.pickle", "wb") as p:
            pickle.dump(self.ParamFiledata, p)
        p.close()

    def get_average_shape(self,):
        h_size = 0
        v_size = 0
        for img in self.initial_files:
            im = cv2.imread(img)
            h_size+=im.shape[1]
            v_size+=im.shape[0]


        self.avg_shape = (v_size//self.num_initial_files, h_size//self.num_initial_files)

        self.save_average_shape()




    def augment_data(self, use_brightness_list = False):
        counter = 0
        if use_brightness_list:
            for img in tqdm(self.initial_files):
                img_ = cv2.imread(img)
                for b in self.brightness_list:
                    out_img = self.controller(img_, brightness = b, contrast = self.contrast_list[0])
                    cv2.imwrite(f"{self.folder_path}/{counter}.jpeg", out_img)
                    counter+=1

        # refreshing the number of files in the directory list
        self.get_files_in_directory()

        idx = 0
        for i in tqdm(range(self.number_of_files_required - self.num_initial_files)):
            new_file_name = self.initial_files[idx].replace(".jpeg", '')
            shutil.copy(self.initial_files[idx], f"{new_file_name}-{counter}.jpeg")
            counter+=1
            idx+=1
            if idx>=self.num_initial_files:
                idx = 0


        # Calculating the average shape of the training data
        self.get_average_shape()





# OPTION 3: CREATE TRAINING DATA USING THE EXTRACTED PHOTOS
# REBUILD_DATA =  False
class Data_creator():
    def __init__(self, AVG_SHAPE = (32,20), training_data_parent_path = "../Training/training_images/", save_labels = False, output_file = "TrainingData_v2.npy"):
        self.AVG_SHAPE = AVG_SHAPE
        self.parent_path = training_data_parent_path
        self.LABELS = {}
        self.save_file = "TDATsave.pickle"
        self.output_file = output_file
        self.training_data = []

        
        # save the labels in the TDATsave.pickle file if the user overrides it to True
        if save_labels or not os.path.exists(self.save_file):
            print("Creating Labels dictionary and saving to TDATsave.pickle file...")
            self.label_folders = sorted([int(i) for i in os.listdir(self.parent_path)])

            # NOTE: Loops to initialize some variables above
            for label_idx in tqdm(range(len(self.label_folders))):
                self.LABELS[str(self.label_folders[label_idx])] = label_idx
            with open(self.save_file, 'wb') as f:
                pickle.dump(self.LABELS, f)
        else:
            print("Loading existing labels from TDATsave.pickle file...")
            with open(self.save_file, 'rb') as f:
                self.LABELS = pickle.load(f)



    def __len__(self,):
        return len(self.LABELS)


    def make_training_dataset(self,):
        for label in self.LABELS:
            for f in tqdm(os.listdir(self.parent_path+label)):
                try:
                    path = self.parent_path+os.path.join(label, f)
                    # print(path)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = img[ 0:img.shape[0], 0: int(img.shape[1]*0.28)]
                    img = cv2.resize(img, self.AVG_SHAPE)
                    # plt.imshow(img)
                    # plt.show()
                    self.training_data.append(
                        [np.array(img),
                        np.eye(len(self))[self.LABELS[label]]
                        ])
                except:
                    print(f"Error while accessing file:{f}")

        np.random.shuffle(self.training_data)
        np.save(f"{self.parent_path}/{self.output_file}", self.training_data)


# OPTION 0: resetting the ID for now...
def reset_paramDat():
    d = pickle.load(open("paramDat.pickle", 'rb'))
    d['ID'] = 0
    pickle.dump(d, open('paramDat.pickle', 'wb'))


#OPTION 5: Create the question data file

ms_id_to_img = {}
def create_id_to_img_dict(score_to_id_dict, parent_folder, path, img_files):
    global ms_id_to_img
    path_ms = "Lab/ms_extracted"
    id_to_img = {}
    id_used = {}
    for key in score_to_id_dict.keys():
        for id_ in score_to_id_dict[key]:
            files = []
            files_ms = []
            for file in os.listdir(img_files):
                if file.replace(".jpeg", '').split('_')[0]==id_:
                    files.append(file)

            # creating the ms dict
            for file in os.listdir(path_ms):
                if file.replace(".jpeg",'').split('_')[0]==id_:
                    files_ms.append(file)
            
            files = sorted(files)
            files_ms = sorted(files_ms)
            imgs = []
            imgs_ms = []
            for f in files:
                imgs.append(cv2.imread(f"{path}/{parent_folder}/{f}", cv2.IMREAD_GRAYSCALE))
                
            for f in files_ms:
                imgs_ms.append(cv2.imread(f"{path_ms}/{f}",cv2.IMREAD_GRAYSCALE))
            id_to_img[id_] = np.array(imgs)
            ms_id_to_img[id_] = np.array(imgs_ms)
            id_used[id_] = np.array([False, int(key)])

    return id_to_img, id_used


def create_qs_data_file(name_qs, num_papers, qs_target_marks, mcq_target_marks):#name_mcq):    
    global ms_id_to_img
    mcq_score_to_id = {}
    qs_score_to_id = {}
    path = "Lab/extracted"
    for folder in os.listdir(path):
        img_files = os.listdir(f"{path}/{folder}")
        for file in img_files:
            const = file.replace(".jpeg", '').split('_')
            score_ = const[-1]
            id_ = const[0]
            if folder=="mcq":
                if score_ in mcq_score_to_id.keys():
                    if id_ not in mcq_score_to_id[score_]:
                        mcq_score_to_id[score_].append(id_)
                else:
                    mcq_score_to_id[score_] = [id_]
            else:
                
                if score_ in qs_score_to_id.keys():
                    if id_ not in qs_score_to_id[score_]:
                        qs_score_to_id[score_].append(id_)
                else:
                    qs_score_to_id[score_] = [id_]


    qs_id_to_img, qs_id_used= create_id_to_img_dict(qs_score_to_id, 'qs', path, "Lab/extracted/qs")
    
    if len(mcq_score_to_id):
        mcq_id_to_img, mcq_id_used= create_id_to_img_dict(mcq_score_to_id, 'mcq', path, "Lab/extracted/mcq")
    else:
        mcq_id_to_img, mcq_id_used = [None, None]
    data_qs = [qs_score_to_id, qs_id_to_img, qs_id_used,mcq_score_to_id, mcq_id_to_img, mcq_id_used, num_papers,ms_id_to_img, qs_target_marks, mcq_target_marks]
   
    pickle.dump(data_qs, open(name_qs, 'wb'))
    del data_qs
      