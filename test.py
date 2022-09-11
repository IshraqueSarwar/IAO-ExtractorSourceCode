
'''
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
import time


import torch
from classifier import NET, NET_


# min: 5800
# max: 7800


# OPTION 1: Training DATA Extractor from pdf
class Question_Extractor():
    def __init__(self, folder_path, h_rm, v_rm, reset_save = False, min_threshold_area = 5000, max_threshold_area = 7000, crp_pct = 0.7,keep_raw_photo_for_check = True, remove_front = 1, remove_back = 1):
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
        out = self.differentiator(d_t)+1
        cv2.imwrite(f'{self.folder_path}/training_images/{out}_{self.name_counter}.jpeg', training_dat)#(f'{self.folder_path}/training_images/{self.dat_num}_detected.jpeg', training_dat)#
        self.name_counter+=1

        # self.dat_num+=1





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


  
                if self.is_mark(r_t):
                    has_mark = True



                    cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 0, 255), 2)
                    
                    if extract_t_d:
                        self.extract_training_data(d_t, training_dat)
                    else:
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

            

            
            # self.bottom = img.shape[0]


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




if __name__ == "__main__":
    pdfs = glob.glob("Lab/*.pdf")
    for pdf_name in pdfs:

        satisfied = False
        # create_training_file = False
        while not satisfied:
            h_rm =int(input("h_rm: ")) #70
            v_rm = int(input('v_rm: ')) #100
            try:
                Extractor = rewiew_pages("../Training", h_rm, v_rm)
                Extractor.convertPdf2Photos(pdf_name, n_page  = 4)
                Extractor.crop_save()
                satisfied_user = input("Are you satisfied with the crop(y/n): ")
                if satisfied_user.lower() == 'y' or satisfied_user.lower() == 'yes':
                    satisfied = True
                # create_t_files_prompt = input("Do you want to create Training Files(y/n): ")
                # if create_t_files_prompt.lower()=='y' or create_t_files_prompt.lower() == "yes":
                #     create_training_file = True
            except:
                print("\n!! ERROR !!\n\n")

        remove_back = int(input("Enter no. pages to remove from back: "))
        Extractor = Question_Extractor("../Training", h_rm, v_rm,keep_raw_photo_for_check = True,remove_back = remove_back)#min_threshold_area = 2000, max_threshold_area =10000, crp_pct = 0.4)#,keep_raw_photo_for_check = True)
        Extractor.convertPdf2Photos(pdf_name)
        Extractor.crop_save()
        Extractor.extract_data(extract_t_d = False)

'''



import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle

# mcq_score = []
# mcq_score_to_id = {}

# qs_score = []
# qs_score_to_id = {}
# name_qs = "question_datafiles/Chemistry_unit-1_qs.pickle"
# name_mcq = "question_datafiles/Chemistry_unit-1_mcq.pickle"
# path = "Lab/extracted"
# # img_files=[]
# total_mark = 0
# for folder in os.listdir(path):
#     img_files = os.listdir(f"{path}/{folder}")
#     # print(len(img_files))
#     for file in img_files:
#         const = file.replace(".jpeg", '').split('_')
#         score_ = const[-1]
#         id_ = const[0]
#         # print(id_)
#         if folder=="mcq":
#             mcq_score.append(score_)
#             if score_ in mcq_score_to_id.keys():
#                 if id_ not in mcq_score_to_id[score_]:
#                     mcq_score_to_id[score_].append(id_)
#             else:
#                 mcq_score_to_id[score_] = [id_]
#         else:
#             qs_score.append(score_)
#             if score_ in qs_score_to_id.keys():
#                 if id_ not in qs_score_to_id[score_]:
#                     qs_score_to_id[score_].append(id_)
#             else:
#                 qs_score_to_id[score_] = [id_]


# # print(qs_score_to_id)


# def create_id_to_img_dict(score_to_id_dict, parent_folder):
#     id_to_img = {}
#     id_used = {}
#     for key in score_to_id_dict.keys():
#         for id_ in score_to_id_dict[key]:
#             files = []
#             for file in img_files:
#                 if file.replace(".jpeg", '').split('_')[0]==id_:
#                     files.append(file)

#             files = sorted(files)
#             imgs = []
#             for f in files:
#                 imgs.append(np.asarray(cv2.imread(f"{path}/{parent_folder}/{f}")))
#             id_to_img[id_] = imgs
#             id_used[id_] = False
#     return id_to_img, id_used

# qs_id_to_img, qs_id_used = create_id_to_img_dict(qs_score_to_id, 'qs')
# mcq_id_to_img, mcq_id_used = create_id_to_img_dict(mcq_score_to_id, 'mcq')

# data_mcq = [mcq_score_to_id, mcq_id_to_img,]
# data_qs = [qs_score_to_id]#, qs_score_to_id, qs_id_to_img]
# pickle.dump(data_qs, open(name_qs, 'wb'))
# # pickle.dump(data_mcq, open(name_mcq, 'wb'))




# ar = pickle.load(open("question_datafiles/Chemistry_unit-1_qs.pickle", 'rb'))

# actual_score = []
# score_dict_ = {}
# for key in ar[0].keys():
#     score_dict_[int(key)] = len(ar[0][key])
#     for i in range(len(ar[0][key])):
#         actual_score.append(int(key))


# candidates = sorted(actual_score)

# res = []
# target = 60

# def backtrack(cur, pos, target):
#     if target == 0:
#         res.append(cur.copy())
#     if target<=0:
#         return

#     prev = -1
#     for i in range(pos, len(candidates)):
#         if candidates[i] == prev:
#             continue
#         cur.append(candidates[i])
#         backtrack(cur, i+1, target - candidates[i])
#         cur.pop()
#         prev = candidates[i]

# backtrack([], 0, target)
# np.random.shuffle(res)
# # print(res)

# combo_list = []

# # pos = 100

# combo = []
# for pos in range(len(res)-1):
#     score_dict = score_dict_.copy()
#     combo = []
#     for i in range(pos, len(res)):
#         use_combo = True
#         # print(score_dict)
#         for x in range(len(res[i])):
#             score = res[i][x]
#             if score_dict[score]>0:
#                 score_dict[score]-=1
#             else:
#                 use_combo = False
#                 for c in range(x):
#                     score_dict[res[i][c]]+=1
#                 break
#         if use_combo:
#             combo.append(res[i])
        

#         if len(combo)==10 or pos>=len(res):
#             break
#     if len(combo) ==10:
#         combo_list.append(combo)
# print(combo_list)
# print(len(combo_list))


# ms_id_to_img = {}

import matplotlib.pyplot as plt
import pickle
import numpy as np
# def create_id_to_img_dict(score_to_id_dict, parent_folder, path, img_files):
#     global ms_id_to_img
#     path_ms = "Lab/ms_extracted"
#     id_to_img = {}
#     id_used = {}
#     for key in score_to_id_dict.keys():
#         for id_ in score_to_id_dict[key]:
#             files = []
#             files_ms = []
#             for file in img_files:
#                 if file.replace(".jpeg", '').split('_')[0]==id_:
#                     files.append(file)

#             # creating the ms dict
#             for file in os.listdir(path_ms):
#                 if file.replace(".jpeg",'').split('_')[0]==id_:
#                     files_ms.append(file)
            
#             files = np.array(sorted(files))
#             files_ms = np.array(sorted(files_ms))
#             imgs = []
#             imgs_ms = []
#             for f in files:
#                 imgs.append(cv2.imread(f"{path}/{parent_folder}/{f}", cv2.IMREAD_GRAYSCALE))
#             for f in files_ms:
#                 imgs_ms.append(cv2.imread(f"{path_ms}/{f}",cv2.IMREAD_GRAYSCALE))
#             id_to_img[id_] = np.array(imgs)
#             ms_id_to_img[id_] = np.array(imgs_ms)
#             id_used[id_] = np.array([False, int(key)])

#     return id_to_img, id_used


# def create_qs_data_file(name_qs, num_papers):#name_mcq):    
#     global ms_id_to_img
#     mcq_score = np.array([])
#     mcq_score_to_id = {}
#     qs_score = np.array([])
#     qs_score_to_id = {}
#     path = "Lab/extracted"

#     for folder in os.listdir(path):
#         img_files = os.listdir(f"{path}/{folder}")
#         # print(len(img_files))
#         for file in img_files:
#             const = file.replace(".jpeg", '').split('_')
#             score_ = const[-1]
#             id_ = const[0]
#             # print(id_)
#             if folder=="mcq":
#                 np.append(mcq_score,score_)
#                 if score_ in mcq_score_to_id.keys():
#                     if id_ not in mcq_score_to_id[score_]:
#                         mcq_score_to_id[score_].append(id_)
#                 else:
#                     mcq_score_to_id[score_] = [id_]
#             else:
#                 np.append(qs_score,score_)
#                 if score_ in qs_score_to_id.keys():
#                     if id_ not in qs_score_to_id[score_]:
#                         qs_score_to_id[score_].append(id_)
#                 else:
#                     qs_score_to_id[score_] = [id_]




#     qs_id_to_img, qs_id_used= create_id_to_img_dict(qs_score_to_id, 'qs', path, img_files)
#     if len(mcq_score):
#         mcq_id_to_img, mcq_id_used= create_id_to_img_dict(mcq_score_to_id, 'mcq', path, img_files)
#     else:
#         mcq_id_to_img, mcq_id_used = [None, None]
#     data_qs = [qs_score_to_id, qs_id_to_img, qs_id_used,mcq_score_to_id, mcq_id_to_img, mcq_id_used, num_papers,ms_id_to_img]
#     # data_qs = [qs_id_to_img, qs_id_used,mcq_id_to_img, mcq_id_used, num_papers, ms_id_to_img ]
    
#     pickle.dump(data_qs, open(name_qs, 'wb'))
    # del data_qs
# create_qs_data_file("question_datafiles/Physics_paper_1.pickle", 12)




# d = pickle.load(open("question_datafiles/Physics_paper_1.pickle",'rb'))
# total = 0



# ids = [5,10,55,65,95,135,140,160,170]
# ids = [str(i) for i in ids]

d= pickle.load(open("question_datafiles/test.pickle", 'rb'))
qs_id_to_img = d[1]
mcq_id_to_img = d[4]
ms_id_to_img = d[-3]

l = len(qs_id_to_img)
if mcq_id_to_img:
    l+=len(mcq_id_to_img)

print(len(ms_id_to_img))
print(l)


total = 0
for score in d[0].keys():
    total+=int(score)*len(d[0][score])


for score in d[3].keys():
    total+=int(score)*len(d[3][score])

print(total)

# for i in mcq_id_to_img:
#     plt.imshow(mcq_id_to_img[i][0])
#     plt.show()
#     plt.imshow(ms_id_to_img[i][0])
#     plt.show()
