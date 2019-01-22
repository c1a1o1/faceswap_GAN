
import shutil
import os
import stat
import time
import sys

def start(video_path_A,video_path_B,output_video_path_A,output_video_path_B):


    #delete
    def remove_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    def remove_dir_tree(remove_dir):
        try:
            shutil.rmtree(remove_dir, ignore_errors=False, onerror=remove_readonly)
        except(PermissionError) as e:  ## if failed, report it back to the user ##
            print("[Delete Error] %s - %s." % (e.filename,e.strerror))
    try:
        remove_dir_tree("./faceA")
        remove_dir_tree("./faceB")
        #remove_dir_tree("./models")
        os.remove("faceA_dummy.mp4")
        os.remove("faceB_dummy.mp4")
    except(FileNotFoundError) :
        pass
    print("remove file")
    #Video A save
    import mtcnn_control_ho as face_detecter
    face_detection= face_detecter.MTCNN_video_face()
    face_detection.setMedia_source_path(video_path_A)
    face_detection.setFaces_path("./faceA")
    face_detection.setOutput_file_path("faceA_dummy.mp4")
    face_detection.start()

    #Video B save
    face_detection.setMedia_source_path(video_path_B)
    face_detection.setFaces_path("./faceB")
    face_detection.setOutput_file_path("faceB_dummy.mp4")
    face_detection.start()
    time.sleep(3)
    #os.system("conda activate faceswap && cd C:/Users/kyu/Desktop/Cox/FaceSwap/gan_repack && "
     #         "python prep_binary_masks_ho.py")


    #os.system("conda activate faceswap && cd C:/Users/kyu/Desktop/Cox/FaceSwap/gan_repack && "
      #        "python train_modul_ho.py 40000")


    #os.system("conda activate faceswap && cd C:/Users/kyu/Desktop/Cox/FaceSwap/gan_repack && "
       #       "pythoon video_conversin_ho.py {A_path} {B_path}".format(A_path  = video_path_A,B_path = video_path_B))
    #preprocessing

    # import prep_binary_masks_ho
    # prep_binary_masks_ho.prep_masks()
    # time.sleep(3)

    #train modul
    # import train_modul_ho as tm
    # tm.train(10);
    # time.sleep(3)
    #conversion video make OUTPUT_VIDEO.mp4 in this dir
    # print("Making Video")
    # import video_conversion_ho as mkvideo
    # mkvideo.conversion(video_path_A,video_path_B,output_video_path_A,output_video_path_B)
    # time.sleep(3)
    # return output_video_path_A,output_video_path_B;


if (len(sys.argv)<3):
    start("C:/Users/kyu/Desktop/1.mp4","C:/Users/kyu/Desktop/2.mp4","AOut.mp4","BOut.mp4")
else :
    start(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])