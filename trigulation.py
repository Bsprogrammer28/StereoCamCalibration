import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Import all calibration parameters
left_camera_calibration = np.load('left_camera_calibration.npz')
right_camera_calibration = np.load('right_camera_calibration.npz')
streo_calibration = np.load('stereo_camera_calibration.npz')

left_camera_matrix = left_camera_calibration['mtx']
right_camera_matrix = right_camera_calibration['mtx']

rotational_matrix = streo_calibration['R']
translation_matrix = streo_calibration['T']

def triangulate_points(points_arr_left, points_arr_right):
    uvs1 = points_arr_left
    uvs2 = points_arr_right
    
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)
    
    
    frame1 = cv.imread('images/imgL.png')
    frame2 = cv.imread('images/imgR.png')
    
    plt.imshow(frame1[:,:,[2,1,0]])
    plt.scatter(uvs1[:,0], uvs1[:,1])
    plt.show()
    
    plt.imshow(frame2[:,:,[2,1,0]])
    plt.scatter(uvs2[:,0], uvs2[:,1])
    plt.show()

    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = left_camera_matrix @ RT1 #projection matrix for C1
    
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([rotational_matrix, translation_matrix], axis = -1)
    P2 = right_camera_matrix @ RT2 #projection matrix for C2

    def DLT(P1, P2, point1, point2):
    
        A = [point1[1]*P1[2,:] - P1[1,:],
            P1[0,:] - point1[0]*P1[2,:],
            point2[1]*P2[2,:] - P2[1,:],
            P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        #print('A: ')
        #print(A)
    
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
    
        print('Triangulated point: ')
        print(Vh[3,0:3]/Vh[3,3])
        return Vh[3,0:3]/Vh[3,3]

    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-15, 5)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(10, 30)
    
    connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
    for _c in connections:
        print(p3ds[_c[0]])
        print(p3ds[_c[1]])
        ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
    
    plt.show()

if __name__ == "__main__":
    # Example points in the left and right images
    uvs1 = [[458, 86], [451, 164], [287, 181],
        [196, 383], [297, 444], [564, 194],
        [562, 375], [596, 520], [329, 620],
        [488, 622], [432, 52], [489, 56]]
    
    uvs2 = [[540, 311], [603, 359], [542, 378],
            [525, 507], [485, 542], [691, 352],
            [752, 488], [711, 605], [549, 651],
            [651, 663], [526, 293], [542, 290]]
    
    triangulate_points(uvs1, uvs2)