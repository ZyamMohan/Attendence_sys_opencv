# importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec

# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

syam = face_rec.load_image_file('sample_photos/syam.jpg')
syam = cv2.cvtColor(syam, cv2.COLOR_BGR2RGB)
syam = resize(syam, 0.50)

syam_test = face_rec.load_image_file('sample_photos/syam_test.jpg')
syam_test = resize(syam_test, 0.50)
syam_test = cv2.cvtColor(syam_test, cv2.COLOR_BGR2RGB)

# finding face location

faceLocation_syam = face_rec.face_locations(syam)[0]
encode_syam = face_rec.face_encodings(syam)[0]
cv2.rectangle(syam, (faceLocation_syam[3], faceLocation_syam[0]), (faceLocation_syam[1], faceLocation_syam[2]), (255, 0, 255), 3)

faceLocation_syamtest = face_rec.face_locations(syam_test)[0]
encode_syamtest = face_rec.face_encodings(syam_test)[0]
cv2.rectangle(syam_test, (faceLocation_syam[3], faceLocation_syam[0]), (faceLocation_syam[1], faceLocation_syam[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_syam], encode_syamtest)
print(results)
cv2.putText(syam_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', syam)
cv2.imshow('test_img', syam_test)
cv2.waitKey(0)
cv2.destroyAllWindows()