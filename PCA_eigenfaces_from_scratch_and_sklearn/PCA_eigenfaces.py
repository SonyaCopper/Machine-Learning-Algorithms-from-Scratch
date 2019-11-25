
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

images=np.array([cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in glob.glob(r"PATH_TO_YOUR_IMAGES")])
img=cv2.imread(r'PATH_TO_YOUR_TEST_IMAGE',cv2.IMREAD_GRAYSCALE)


def reshape(images):
    new_images=np.zeros((images.shape[1]*images.shape[2],images.shape[0]))
    for i in range(images.shape[0]):
        new_images[:,i]=images[i].flatten()
    return new_images

def sort_eigenvectors(gamma,v):
    ix=np.argsort(gamma)[::-1]
    v_sorted=(v.T[ix]).T
    return v_sorted

def find_w(eigenfaces,img):
    w=np.zeros(eigenfaces.shape[0])
    for i in range(eigenfaces.shape[0]):
        w[i]=img.flatten().dot(eigenfaces[i].flatten())  
    return w

def reconstruct_face(w,eigenfaces):
    e=np.zeros(eigenfaces.shape)
    for  i in range(len(w)):
        e[i]=w[i]*eigenfaces[i]
    face=np.sum(e,axis=0)
    return face
    
def get_eigenfaces(images,v):
    eigenvectors=np.zeros((components,images.shape[0]))
    for i in range(components):
        u=images.dot(v.T[i])
        eigenvectors[i]=u/np.linalg.norm(u)
    return eigenvectors

im,x,y=images.shape
mean_img=np.mean(images,axis=0)
images=images-mean_img
components=30
img_matrix=StandardScaler().fit(reshape(images)).transform(reshape(images))



########################################### PCA eigenfaces sklearn ##################################
pca=PCA(n_components=components).fit(img_matrix.T)
eigenfaces_pca= pca.components_.reshape((components, x, y))
w_pca=find_w(eigenfaces_pca,img-mean_img)
rec_face=reconstruct_face(w_pca,eigenfaces_pca)+mean_img 

########################################### PCA eigenfaces implementation ##################################

s=np.cov(img_matrix.T)
gamma,v=np.linalg.eig(s)
v_sorted=sort_eigenvectors(gamma,v)  
eigenfaces= get_eigenfaces(img_matrix,v_sorted).reshape((components,x, y))
w=find_w(eigenfaces,img-mean_img)
rec_image=reconstruct_face(w,eigenfaces)+mean_img



plt.figure(1)
plt.imshow(img)
plt.figure(2)
plt.imshow(rec_face)
plt.figure(3) 
plt.imshow(rec_image)  






















