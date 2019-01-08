import os
print(os.path)
#os.system("conda activate faceswap && cd C:/Users/kyu/Desktop/Cox/FaceSwap/gan_repack && python train_modul_ho.py 40000")
import sys
os.system("conda activate faceswap && cd C:/Users/kyu/Desktop/Cox/FaceSwap/gan_repack && python video_conversion_ho.py "+sys.argv[1]+" "+sys.argv[2])
os.system("conda activate faceswap && cd C:/Users/kyu/Desktols/Cox/FaceSwap/gan_repack && python video_conversion_ho.py "+sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]+" "+sys.argv[4])